# MatClaw 在 vepfs + Docker 环境下的排障与环境指南

本文档记录了在 vepfs 网络文件系统服务器上，通过 SSH 代理转发和 Docker 容器运行 MatClaw 的完整调试过程。供未来的开发者和 AI 编程助手参考。

## 服务器环境概览

| 项目 | 详情 |
|------|------|
| **操作系统** | Linux 5.4.250（Debian 系） |
| **文件系统** | vepfs（网络文件系统，挂载于 `/vepfs-mlp2/`） |
| **Docker** | 客户端 28.2.2，服务端 24.0.9（版本不一致，API 降级至 1.43） |
| **Docker 根目录** | `/ebs/docker/165536.165536`（不在 vepfs 上） |
| **GPU** | NVIDIA（驱动 535.129.03，CUDA 12.2），通过 `--gpus all` 传入容器 |
| **代理** | SSH 端口转发 → `127.0.0.1:7890`，通过 `~/.bashrc` 中的 `http_proxy`/`https_proxy` 暴露 |
| **认证** | Claude OAuth，凭据存于 `/root/.claude/.credentials.json`（`.env` 中无 API key） |
| **Node.js** | 安装在 vepfs 上 `/vepfs-mlp2/mlp-public/250266/` |

## 架构概览

```
飞书/Web → MatClaw 宿主机进程（Node.js）
                ↓ 为每个聊天组启动 Docker 容器
            Docker 容器（matclaw-agent:cuda）
                ↓ 运行 agent-runner → Claude Agent SDK
                ↓ 通过 stdout（主路径）+ IPC 文件（备用路径）输出结果
            MatClaw 宿主机进程
                ↓ 将回复发送到飞书/Web
```

涉及的关键文件：
- `src/container-runner.ts` — 启动容器、读取输出、管理挂载
- `container/agent-runner/src/index.ts` — 在容器内运行，调用 Claude SDK
- `container/Dockerfile` — 容器镜像构建（entrypoint 在启动时编译 TypeScript）

---

## 问题 1：容器启动即崩溃（ENOENT）

### 现象
容器约 4 秒后以 exit code 1 退出。stdout 和 stderr 均为空，无任何错误信息。

### 根因
Claude Agent SDK 启动时调用 `appendFileSync('~/.claude/debug/<uuid>.txt')` 写调试日志，但它**不会自动创建 `debug/` 目录**。MatClaw 为每个聊天组创建的 `.claude/` 挂载目录中只有 `settings.json` 和 `skills/`，没有 `debug/`。

```
容器内部发生了什么：
1. entrypoint.sh 编译 TypeScript ✓
2. 启动 node agent-runner ✓
3. agent-runner 调用 Claude SDK
4. SDK 执行 appendFileSync('/home/node/.claude/debug/xxx.txt')
5. 目录不存在 → 抛出 ENOENT 异常 → 进程崩溃
```

### 排查过程
1. 容器日志显示 stdout/stderr 为空 —— 从外部看不到任何线索
2. 用 `--entrypoint /bin/bash` 手动运行容器，逐步调试
3. 将 node 的 stderr 重定向到挂载卷中的文件，终于捕获到了真正的错误：
   ```
   Error: ENOENT: no such file or directory, open '/home/node/.claude/debug/xxx.txt'
   ```

### 修复
在 `src/container-runner.ts` 中，创建 `groupSessionsDir`（即 `.claude/` 挂载目录）之后：

```typescript
mkdirWorld(path.join(groupSessionsDir, 'debug'));
```

### 教训
- 当容器的 stdout/stderr 为空时，不代表"什么都没发生"，可能是进程在产生输出之前就崩了
- 通过挂载卷写文件是在 Docker stdout 不可用时获取调试信息的可靠手段

---

## 问题 2：网络文件系统的权限映射（EACCES）

### 现象
修复问题 1 后，错误变为 `EACCES: permission denied`，同一个路径。

### 根因
vepfs 的 UID 映射机制与本地文件系统不同。宿主机以 root（uid 0）创建的目录，在容器内看到的 owner 变成了 `nobody:nogroup`。容器内的 `node` 用户（uid 1000）在默认 `755` 权限下无法写入。

```
宿主机（root, uid=0）创建目录，权限 755
    ↓ 通过 -v 挂载到容器
容器内看到 owner 是 nobody:nogroup（vepfs UID 映射）
    ↓
容器内的 node 用户（uid=1000）→ 无写权限（只有 owner 有写权限）
```

受影响的目录包括所有容器需要写入的目录：
- `~/.claude/` 及 `~/.claude/debug/`
- `/workspace/group/`（工作目录）
- `/workspace/ipc/` 及所有子目录（`messages/`、`tasks/`、`input/`、`output/`）
- 其他所有 bind mount 的可写路径

### 修复
在 `src/container-runner.ts` 中创建了一个辅助函数：

```typescript
/**
 * 创建目录并设置为全局可写。
 * 在网络文件系统（vepfs、NFS）上，宿主机以 root 创建的目录，
 * 容器内可能映射为 nobody:nogroup，需要 0o777 权限确保容器用户可写。
 */
function mkdirWorld(dirPath: string): void {
  fs.mkdirSync(dirPath, { recursive: true });
  try { fs.chmodSync(dirPath, 0o777); } catch { /* 尽力而为 */ }
}
```

将所有容器可写目录的 `fs.mkdirSync(...)` 替换为 `mkdirWorld(...)`。

对于已有的安装，还需要一次性修复权限：
```bash
chmod -R 777 /path/to/matclaw/data/ /path/to/matclaw/groups/
```

### 教训
- 本地开发环境不会遇到这个问题（UID 一致），只在网络文件系统上出现
- 一次性修权限不够，要在代码中保证**每次创建新目录都设权限**，因为新的聊天组会不断创建新目录

---

## 问题 3：API 返回 403 Forbidden

### 现象
容器不再崩溃，SDK 正常启动，但 Claude API 返回 `403 {"error":{"type":"forbidden","message":"Request not allowed"}}`。

### 根因
宿主机通过 SSH 端口转发做代理（`http_proxy=http://127.0.0.1:7890`），宿主机的 `claude` 命令能正常工作。但 Docker 容器是**独立的网络环境**，不继承宿主机的环境变量：

```
宿主机的 claude 命令 → 读 $http_proxy → 走 127.0.0.1:7890 → 代理转发 → API ✓
容器内的 claude     → 没有 $http_proxy → 直连 api.anthropic.com → 被拒 ✗
```

### 修复
在 `src/container-runner.ts` 的 `buildContainerArgs()` 函数中，自动将宿主机的代理环境变量传入容器：

```typescript
for (const proxyVar of [
  'http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY',
  'no_proxy', 'NO_PROXY',
]) {
  const val = process.env[proxyVar];
  if (val) {
    // 将 127.0.0.1/localhost 替换为 host.docker.internal
    const containerVal = val.replace(
      /127\.0\.0\.1|localhost/g,
      'host.docker.internal',
    );
    args.push('-e', `${proxyVar}=${containerVal}`);
  }
}
// Linux 上需要显式添加 host.docker.internal 解析
if (process.platform === 'linux' && hasProxy) {
  args.push('--add-host', 'host.docker.internal:host-gateway');
}
```

**关键点：** `127.0.0.1` 在容器内指向容器自己，不是宿主机。必须替换为 `host.docker.internal`。

### 教训
- Docker 容器是隔离环境：网络、环境变量、文件系统都是独立的
- `host.docker.internal` 是 Docker 提供的特殊域名，用于容器内访问宿主机
- 在 Linux 上使用 `host.docker.internal` 需要加 `--add-host host.docker.internal:host-gateway`

---

## 问题 4：ECONNREFUSED 172.17.0.1:7890

### 现象
代理环境变量传进去了，但容器报 `connect ECONNREFUSED 172.17.0.1:7890`。

### 根因
SSH 端口转发默认只绑定 `127.0.0.1`。Docker 的 `host.docker.internal` 解析为 Docker 网桥 IP `172.17.0.1`，而这个 IP 上没有监听 7890 端口。

```
┌─────────────────────────────────────────────┐
│ 宿主机                                       │
│                                              │
│   SSH 转发 → 127.0.0.1:7890（只监听本地回环） │
│                                              │
│   Docker 网桥 172.17.0.1 ←─ 容器通过这里访问  │
│        ↑ 这里没有监听 7890！                  │
└─────────────────────────────────────────────┘
```

可以通过 `ss -tlnp | grep 7890` 确认：
```
LISTEN  127.0.0.1:7890   ← SSH 转发（只听本地）
# 172.17.0.1:7890 上什么都没有
```

### 修复
用 `socat` 做端口桥接：

```bash
socat TCP-LISTEN:7890,fork,reuseaddr,bind=172.17.0.1 TCP:127.0.0.1:7890 &
```

这条命令的意思是：在 `172.17.0.1:7890` 监听，收到的连接全部转发到 `127.0.0.1:7890`。

```
容器 → 172.17.0.1:7890 → socat 桥接 → 127.0.0.1:7890 → SSH 转发 → 外网
```

**安全性：** 只监听 Docker 网桥 IP，不监听 `0.0.0.0`，外部无法访问。

每次 SSH 重连后需要重启 socat。已添加到 `~/.bashrc` 自动启动：

```bash
# 桥接 SSH 代理到 Docker 网桥，使容器可以使用代理
if ! ss -tln 2>/dev/null | grep -q '172.17.0.1:7890'; then
  socat TCP-LISTEN:7890,fork,reuseaddr,bind=172.17.0.1 TCP:127.0.0.1:7890 &>/dev/null &
fi
```

### 教训
- SSH 端口转发默认绑定 `127.0.0.1`，其他 IP（包括 Docker 网桥）访问不到
- `socat` 是解决此类端口桥接问题的瑞士军刀，简单且可靠
- 用 `ss -tlnp | grep <端口>` 可以快速确认谁在监听、监听在哪个 IP

---

## 问题 5：Docker stdout 管道数据丢失

### 现象
API 调通了，Agent 成功生成了回复（在 session 文件中可以看到完整的回答），但 MatClaw 宿主机进程收不到任何输出，飞书也收不到回复。`container-live.log` 始终为空。

### 根因
Docker 在 vepfs 环境下，容器的 **stdout/stderr 管道数据完全丢失**。这是 Docker daemon（运行在 `/ebs/`）与 vepfs 文件系统交互的兼容性问题。

通过多次测试确认：

```bash
# 在这个环境下，任何容器的 stdout 都无法被宿主机捕获：
docker run --rm ubuntu:22.04 echo "hello"
# （空输出）

# 但通过挂载卷写文件是正常的：
docker run --rm -v /vepfs-path:/mnt --entrypoint bash image \
  -c "echo hello > /mnt/test.txt"
# /mnt/test.txt 内容正常
```

`container-runner.ts` 依赖 `container.stdout.on('data')` 事件来接收 agent 输出，但这个事件**永远不会触发**。

### 修复
添加了 **IPC 文件 fallback** 机制 —— 双通道输出。

**容器内（`container/agent-runner/src/index.ts`）：**

```typescript
const IPC_OUTPUT_DIR = '/workspace/ipc/output';
let ipcOutputSeq = 0;

function writeOutput(output: ContainerOutput): void {
  // 主路径：stdout 标记（给正常环境用）
  console.log(OUTPUT_START_MARKER);
  console.log(JSON.stringify(output));
  console.log(OUTPUT_END_MARKER);

  // 备用路径：写 IPC 文件（给 Docker stdout 损坏的环境用）
  try {
    fs.mkdirSync(IPC_OUTPUT_DIR, { recursive: true });
    const seq = String(ipcOutputSeq++).padStart(6, '0');
    fs.writeFileSync(
      path.join(IPC_OUTPUT_DIR, `${seq}-${Date.now()}.json`),
      JSON.stringify(output),
    );
  } catch { /* 尽力而为 */ }
}
```

**宿主机（`src/container-runner.ts`）：**

添加了轮询循环，每 500ms 检查 `ipc/output/` 目录是否有新的 JSON 文件，读取后删除，并注入到与 stdout 相同的输出处理链：

```typescript
const pollIpcOutput = () => {
  if (!ipcPolling) return;
  try {
    const files = fs.readdirSync(ipcOutputDir)
      .filter(f => f.endsWith('.json')).sort();
    for (const file of files) {
      const parsed = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
      fs.unlinkSync(filePath); // 消费文件
      // ... 注入到输出处理链（与 stdout 路径相同）
    }
  } catch { /* 目录可能还不存在 */ }
  setTimeout(pollIpcOutput, 500);
};
```

容器退出时还会做最后一次轮询，确保不遗漏。

### 教训
- Docker stdout 管道**不是 100% 可靠的**，在特殊文件系统上可能失效
- 文件系统 IPC（通过挂载卷共享文件）是更可靠的跨 Docker 通信方式
- 好的架构应该有 fallback 机制，不依赖单一通道

---

## 问题 6：`docker exec` 和 `docker run` 的 stdout 完全静默

### 现象
在宿主机上对运行中的容器执行命令、或启动一次性容器时，**所有 stdout 输出全部丢失** —— 终端上看不到任何内容：

```bash
# 以下命令在宿主机上全部产生零输出：
docker run --rm ubuntu:24.04 echo "hello"
docker exec <运行中的容器> pip list
docker exec <运行中的容器> /bin/bash -c "echo test"
docker exec <运行中的容器> python -c "print('hi')"

# 管道、tee、变量捕获 —— 全部为空：
result=$(docker exec <容器> echo hello); echo "结果: $result"
# 结果:
```

stderr 同样受影响。退出码（exit code）能正确返回（0），但没有任何数据到达。

### 根因
与问题 5 是同一个底层原因。Docker 的 stdout/stderr 管道机制在 Docker daemon 的存储后端（`/ebs/`）和调用进程的工作目录（vepfs）处于不同文件系统类型时，无法将数据传递到宿主机进程。

**受影响的操作：**
- `docker run` — 容器主进程的 stdout/stderr
- `docker exec` — 在已有容器内执行命令的 stdout/stderr
- `docker logs` — 可能也为空或不完整

**不受影响的操作：**
- 容器内部的文件读写（完全正常）
- bind mount 卷上的文件读写（容器写入、宿主机读取 —— 正常）
- `docker cp` — 在容器和宿主机之间复制文件
- `docker inspect`、`docker ps`、`docker history` — 元数据命令正常工作
- 容器内的网络操作（API 调用、数据下载等）

### 解决方法

**方法 1：在容器内写文件，再用 `docker cp` 拷出来**
```bash
# 在容器内执行命令，输出重定向到文件
docker exec <容器> /bin/bash -c "pip list > /tmp/output.txt 2>&1"

# 将文件从容器中拷贝到宿主机
docker cp <容器>:/tmp/output.txt ./output.txt

# 在宿主机上查看
cat ./output.txt
```

**方法 2：写到 bind mount 挂载的卷上**
```bash
# 如果容器已经有挂载卷：
docker exec <容器> /bin/bash -c "pip list > /workspace/group/output.txt 2>&1"

# 直接从宿主机的挂载路径读取
cat groups/<群组>/output.txt
```

**方法 3：用 `docker history` 检查镜像内容**
当需要验证镜像中安装了哪些包（比如 pip 包）时，`docker history` 读取的是镜像元数据而非容器 stdout，因此可以正常工作：

```bash
# 查看镜像中所有 pip install 层：
docker history matclaw-agent:cuda --no-trunc | grep "pip install"
```

### 对 MatClaw 使用的影响

| 场景 | 是否受影响 | 原因 |
|------|-----------|------|
| Agent 执行（飞书发消息触发） | 不受影响 | MatClaw 使用 IPC 文件 fallback（问题 5 的修复） |
| 数据下载（如 matbench） | 不受影响 | 下载操作在容器内部或挂载卷上进行 |
| 模型训练 | 不受影响 | 训练完全在容器内部运行 |
| 手动调试（`docker exec`） | **受影响** | 开发者需要使用上述基于文件的替代方案 |

### 教训
- 在 Docker stdout 损坏的环境下，**永远不要假设命令输出能到达宿主机终端**
- 任何跨容器的通信都应该有基于文件的 fallback 机制
- `docker history` 和 `docker inspect` 是可靠的镜像检查手段，因为它们读取的是 daemon 的元数据，不经过容器 stdout

---

## 调试技巧总结

### 1. Docker stdout 不可用时如何调试
用挂载卷将输出写到文件：

```bash
docker run --rm \
  -v /宿主机路径:/tmp/debug \
  --entrypoint bash 镜像 \
  -c "你的命令 > /tmp/debug/stdout.txt 2>/tmp/debug/stderr.txt"

# 然后在宿主机读取
cat /宿主机路径/stdout.txt
```

### 2. 查看 Claude SDK 调试日志
SDK 会将详细日志写入 `~/.claude/debug/`。查看最新日志：

```bash
# 找到最新的日志文件
ls -lt data/sessions/<group>/.claude/debug/*.txt | head -1

# 过滤关键错误
grep -E "ERROR|403|ECONNREFUSED" <文件路径>
```

### 3. 从外部检查容器内进程
```bash
docker top <容器名>       # 查看容器内运行的进程
docker exec <容器名> bash -c "..."  # 在容器内执行命令（注意 stdout 可能丢失！）
```

### 4. Session 文件包含实际回复
即使输出传递失败，SDK 也会保存对话历史：
```
data/sessions/<group>/.claude/projects/-workspace-group/*.jsonl
```

用以下命令查看 agent 是否真的生成了回复：
```bash
node -e "
const fs=require('fs'), lines=fs.readFileSync('<jsonl文件>','utf8').trim().split('\n');
for(const l of lines){
  const j=JSON.parse(l);
  if(j.type==='assistant') console.log(j.message?.content?.find(c=>c.type==='text')?.text?.slice(0,200));
}"
```

### 5. 端口和进程管理
```bash
ss -tlnp | grep <端口>         # 查看谁在监听某个端口
kill -9 <PID>                   # 强制杀进程（普通 kill 不够时用）
pkill -9 -f "tsx src/index"     # 按命令模式批量杀进程
docker kill $(docker ps -q --filter "name=matclaw")  # 杀所有 matclaw 容器
```

---

## 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `src/container-runner.ts` | 新增 `mkdirWorld()` 辅助函数；创建 `.claude/debug/` 目录；自动转发代理环境变量（localhost→host.docker.internal）；Linux 上添加 `--add-host`；新增 IPC output 文件轮询作为 stdout 的 fallback；所有容器可写目录统一使用 `mkdirWorld()` |
| `container/agent-runner/src/index.ts` | `writeOutput()` 新增 IPC 文件 fallback —— 在写 stdout 的同时，将 JSON 写入 `/workspace/ipc/output/` |
| `~/.bashrc` | 新增 socat 桥接自动启动脚本 |

---

## 类似环境部署检查清单

如果你在网络文件系统 + Docker + SSH 代理的服务器上部署 MatClaw：

- [ ] 确保所有容器可写目录使用 `chmod 777`（或使用 `mkdirWorld()`）
- [ ] 在 `.env` 中设置 `http_proxy`/`https_proxy`，或确保它们在 shell 环境变量中
- [ ] 如果代理只监听 localhost，运行 socat 桥接到 Docker 网桥 IP
- [ ] 验证 Docker stdout 是否正常：`docker run --rm ubuntu echo test` —— 如果为空，需要 IPC fallback
- [ ] 检查 `data/sessions/<group>/.claude/debug/` 中的 SDK 级别错误日志

---

## 概念速查表

| 概念 | 解释 |
|------|------|
| **vepfs** | 一种网络文件系统（类似 NFS），多台机器共享存储，UID 映射可能与本地不同 |
| **Docker 网桥（bridge）** | Docker 默认网络模式，容器通过虚拟网桥（默认 `172.17.0.1`）与宿主机通信 |
| **host.docker.internal** | Docker 提供的特殊域名，在容器内解析为宿主机 IP。Linux 上需要 `--add-host` 才能用 |
| **SSH 端口转发** | 通过 SSH 隧道将远程端口映射到本地，默认只绑定 `127.0.0.1` |
| **socat** | 万能的网络中继工具，可以在任意两个地址之间桥接 TCP/UDP 连接 |
| **IPC（进程间通信）** | 进程之间交换数据的方式，这里指通过共享文件系统（挂载卷）传递 JSON 文件 |
| **chmod 777** | 设置文件/目录为所有用户可读可写可执行，解决跨用户权限问题 |
| **bind mount（-v）** | Docker 将宿主机目录直接挂载到容器内，双方看到同一份数据 |
