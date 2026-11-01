/* NERAL Cloud Coordinator v0: cron heartbeat only.
 * NO GitHub credentials, API jobs, weights, downloads, code execution or paid AI.
 * The separate hourly ChatGPT automation remains the actual PR-development agent.
 */
const json = (body, status = 200) => new Response(JSON.stringify(body), {
  status,
  headers: {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'no-store',
    'x-content-type-options': 'nosniff',
  },
});

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname !== '/health' || url.search) return json({ error: 'not_found' }, 404);
    if (request.method !== 'GET') return json({ error: 'method_not_allowed' }, 405);
    if (!env?.HEARTBEAT || typeof env.HEARTBEAT.get !== 'function') {
      return json({ scheduler: 'unavailable' }, 503);
    }
    try {
      const last = await env.HEARTBEAT.get('last_tick');
      if (last !== null && (typeof last !== 'string' || Number.isNaN(Date.parse(last)))) {
        return json({ scheduler: 'unavailable' }, 503);
      }
      return json({
        service: 'neral-cloud-coordinator',
        scheduler: last ? 'heartbeat_recorded' : 'awaiting_first_tick',
        last_tick: last,
        ai_code_agent_running: false,
        private_github_connected: false,
        gpu_training: false,
        note: 'This heartbeat monitors only the Cloudflare cron. Existing hourly ChatGPT automation handles GitHub Draft PR work separately.',
      });
    } catch {
      return json({ scheduler: 'unavailable' }, 503);
    }
  },
  async scheduled(event, env) {
    if (!env?.HEARTBEAT || typeof env.HEARTBEAT.put !== 'function') {
      throw new Error('HEARTBEAT KV binding is unavailable');
    }
    if (event.cron !== '*/15 * * * *' || !Number.isFinite(event.scheduledTime)) {
      throw new Error('Unexpected or invalid scheduled event');
    }
    await env.HEARTBEAT.put('last_tick', new Date(event.scheduledTime).toISOString());
  },
};
