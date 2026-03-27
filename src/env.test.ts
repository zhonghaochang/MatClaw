import fs from 'fs';
import os from 'os';
import path from 'path';
import { afterEach, describe, expect, it } from 'vitest';

import { getCodexAuthFilePath, readEnvFile } from './env.js';

const originalCwd = process.cwd();
const originalHome = process.env.HOME;
const originalCodexHome = process.env.CODEX_HOME;

const tempDirs: string[] = [];

afterEach(() => {
  process.chdir(originalCwd);

  if (originalHome === undefined) {
    delete process.env.HOME;
  } else {
    process.env.HOME = originalHome;
  }

  if (originalCodexHome === undefined) {
    delete process.env.CODEX_HOME;
  } else {
    process.env.CODEX_HOME = originalCodexHome;
  }

  for (const dir of tempDirs.splice(0)) {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

describe('env helpers', () => {
  it('uses CODEX_HOME for auth.json path when configured', () => {
    process.env.CODEX_HOME = '/tmp/custom-codex-home';
    expect(getCodexAuthFilePath()).toBe('/tmp/custom-codex-home/auth.json');
  });

  it('only auto-detects Claude OAuth when Claude auth keys are requested', () => {
    const projectDir = fs.mkdtempSync(path.join(os.tmpdir(), 'matclaw-env-'));
    const homeDir = fs.mkdtempSync(path.join(os.tmpdir(), 'matclaw-home-'));
    tempDirs.push(projectDir, homeDir);

    fs.writeFileSync(path.join(projectDir, '.env'), 'AGENT_ENGINE=codex\n');
    fs.mkdirSync(path.join(homeDir, '.claude'), { recursive: true });
    fs.writeFileSync(
      path.join(homeDir, '.claude', '.credentials.json'),
      JSON.stringify({
        claudeAiOauth: {
          accessToken: 'test-claude-token',
          expiresAt: Math.floor(Date.now() / 1000) + 3600,
        },
      }),
    );

    process.chdir(projectDir);
    process.env.HOME = homeDir;

    expect(readEnvFile(['AGENT_ENGINE'])).toEqual({ AGENT_ENGINE: 'codex' });
    expect(readEnvFile(['CLAUDE_CODE_OAUTH_TOKEN'])).toMatchObject({
      CLAUDE_CODE_OAUTH_TOKEN: 'test-claude-token',
    });
  });
});
