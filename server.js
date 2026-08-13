import { createServer } from 'node:http';

/**
 * Vercel zero-config Node 서버 진입점.
 *
 * ON-FARM MVP는 로컬 SQLite를 사용한다. Vercel 함수의 파일시스템은 영구 저장소가
 * 아니므로 데모 배포에서는 쓰기 가능한 /tmp 아래에 시드 DB와 업로드를 만든다.
 */
if (process.env.VERCEL) {
  process.env.HOST ??= '0.0.0.0';
  process.env.DATA_DIR ??= '/tmp/onfarm-data';
}

const [{ createApp }, { initProvider }, { db }, { seed }] = await Promise.all([
  import('./dist/server/main.js'),
  import('./dist/ai/providers/index.js'),
  import('./dist/db/index.js'),
  import('./dist/db/seed.js'),
]);

seed(db());
await initProvider();

const app = createApp();
const server = createServer((request, response) => app.emit('request', request, response));
server.listen(Number(process.env.PORT ?? 3000), process.env.HOST ?? '0.0.0.0');
