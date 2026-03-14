/* StudyCore Service Worker — 离线缓存 */
const CACHE = 'studycore-v1';
const STATIC = ['/ui/', '/ui/index.html', '/ui/icon.svg'];

/* 安装：缓存静态资源 */
self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(STATIC)).then(() => self.skipWaiting())
  );
});

/* 激活：清理旧缓存 */
self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

/* fetch：API 请求直接走网络；UI 资源先缓存后网络 */
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);

  /* API / 数据接口 → 总是走网络，不缓存 */
  if (url.pathname.startsWith('/api/')) return;
  if (e.request.method !== 'GET') return;

  e.respondWith(
    caches.match(e.request).then(cached => {
      const networkFetch = fetch(e.request).then(res => {
        /* 只缓存成功的同源响应 */
        if (res.ok && res.type === 'basic') {
          const clone = res.clone();
          caches.open(CACHE).then(c => c.put(e.request, clone));
        }
        return res;
      });
      /* 有缓存就立即返回缓存，同时后台更新 */
      return cached || networkFetch;
    })
  );
});
