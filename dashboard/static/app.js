/* ============================================================
   JARVIS DASHBOARD — App JavaScript
   Particle background, WebSocket, state orb, commands, toasts
   ============================================================ */

// ---------- Utility ----------
function $(sel, ctx = document) { return ctx.querySelector(sel); }
function $$(sel, ctx = document) { return [...ctx.querySelectorAll(sel)]; }
function byId(id) { return document.getElementById(id); }

// ---------- Particle Canvas ----------
class ParticleCanvas {
  constructor(canvasId) {
    this.canvas = byId(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this.particles = [];
    this.mouse = { x: -1000, y: -1000 };
    this.resize();
    this.init();
    window.addEventListener('resize', () => this.resize());
    document.addEventListener('mousemove', (e) => {
      this.mouse.x = e.clientX;
      this.mouse.y = e.clientY;
    });
    this.animate();
  }

  resize() {
    if (!this.canvas) return;
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
  }

  init() {
    this.particles = [];
    const count = Math.min(60, Math.floor((window.innerWidth * window.innerHeight) / 18000));
    for (let i = 0; i < count; i++) {
      this.particles.push({
        x: Math.random() * this.canvas.width,
        y: Math.random() * this.canvas.height,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        r: Math.random() * 1.5 + 0.5,
        alpha: Math.random() * 0.3 + 0.1,
      });
    }
  }

  animate() {
    if (!this.ctx) return;
    const { ctx, canvas, particles, mouse } = this;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0) p.x = canvas.width;
      if (p.x > canvas.width) p.x = 0;
      if (p.y < 0) p.y = canvas.height;
      if (p.y > canvas.height) p.y = 0;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0, 240, 255, ${p.alpha})`;
      ctx.fill();
    }

    // Draw connections
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 120) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(0, 240, 255, ${0.06 * (1 - dist / 120)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }

      // Mouse interaction
      const dx = particles[i].x - mouse.x;
      const dy = particles[i].y - mouse.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 150) {
        ctx.beginPath();
        ctx.moveTo(particles[i].x, particles[i].y);
        ctx.lineTo(mouse.x, mouse.y);
        ctx.strokeStyle = `rgba(139, 92, 246, ${0.12 * (1 - dist / 150)})`;
        ctx.lineWidth = 0.5;
        ctx.stroke();
      }
    }

    requestAnimationFrame(() => this.animate());
  }
}


// ---------- WebSocket Manager ----------
class JarvisWebSocket {
  constructor(onMessage, onStatus) {
    this.onMessage = onMessage;
    this.onStatus = onStatus;
    this.ws = null;
    this.reconnectTimer = null;
    this.connected = false;
    this.connect();
  }

  connect() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    this.ws = new WebSocket(`${proto}//${location.host}/ws`);

    this.ws.onopen = () => {
      this.connected = true;
      this.onStatus(true);
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.onMessage(data);
      } catch (_) {}
    };

    this.ws.onclose = () => {
      this.connected = false;
      this.onStatus(false);
      this.scheduleReconnect();
    };

    this.ws.onerror = () => {
      this.connected = false;
      this.onStatus(false);
    };
  }

  scheduleReconnect() {
    if (this.reconnectTimer) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, 3000);
  }
}


// ---------- Toast System ----------
class ToastManager {
  constructor() {
    this.container = document.createElement('div');
    this.container.className = 'toast-container';
    document.body.appendChild(this.container);
  }

  show(message, duration = 3000) {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    this.container.appendChild(toast);

    setTimeout(() => {
      toast.classList.add('toast-out');
      toast.addEventListener('animationend', () => toast.remove());
    }, duration);
  }
}


// ---------- Smooth Counter ----------
function animateCounter(el, target, duration = 600) {
  if (!el) return;
  const start = parseFloat(el.textContent) || 0;
  const diff = target - start;
  if (diff === 0) return;
  const startTime = performance.now();

  function step(now) {
    const elapsed = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
    const current = start + diff * eased;
    el.textContent = Number.isInteger(target) ? Math.round(current) : current.toFixed(1);
    if (progress < 1) requestAnimationFrame(step);
  }

  requestAnimationFrame(step);
}


// ---------- Sidebar ----------
function initSidebar() {
  const sidebar = $('.sidebar');
  const toggle = byId('sidebar-toggle');
  if (!sidebar || !toggle) return;

  const collapsed = localStorage.getItem('sidebar-collapsed') === 'true';
  if (collapsed) sidebar.classList.add('collapsed');

  toggle.addEventListener('click', () => {
    sidebar.classList.toggle('collapsed');
    localStorage.setItem('sidebar-collapsed', sidebar.classList.contains('collapsed'));
  });

  // Highlight active nav link
  const currentPath = location.pathname;
  $$('.nav-link', sidebar).forEach(link => {
    const href = link.getAttribute('href');
    if (href === currentPath || (href !== '/' && currentPath.startsWith(href))) {
      link.classList.add('active');
    } else if (href === '/' && currentPath === '/') {
      link.classList.add('active');
    }
  });
}


// ---------- State Renderer ----------
function renderState(data) {
  // State orb
  const orbContainer = byId('state-orb-container');
  if (orbContainer) {
    orbContainer.className = `state-orb-container state-${data.state || 'OFFLINE'}`;
  }

  const stateLabel = byId('state-label');
  if (stateLabel) stateLabel.textContent = data.state || 'OFFLINE';

  // Last input/response
  const lastInput = byId('last-input');
  if (lastInput) lastInput.textContent = data.last_input || '';

  const lastResponse = byId('last-response');
  if (lastResponse) {
    lastResponse.textContent = data.last_response || '';
    lastResponse.scrollTop = lastResponse.scrollHeight;
  }

  // Stats
  const fields = {
    'stat-session': data.session_id || 'n/a',
    'stat-model': data.model || 'unknown',
    'stat-uptime': (data.uptime_seconds || 0) + 's',
  };

  for (const [id, val] of Object.entries(fields)) {
    const el = byId(id);
    if (el) el.textContent = val;
  }

  // Animated counters
  const counters = {
    'stat-memory': data.memory_count || 0,
    'stat-goals': data.active_goals || 0,
  };
  for (const [id, val] of Object.entries(counters)) {
    animateCounter(byId(id), val);
  }

  // Ollama indicator
  const ollamaEl = byId('stat-ollama');
  if (ollamaEl) {
    ollamaEl.textContent = data.ollama_online ? 'Online' : 'Offline';
    ollamaEl.className = `stat-value mono ${data.ollama_online ? 'text-success' : 'text-danger'}`;
  }
}


// ---------- Command Form ----------
function initCommandForm() {
  const form = byId('command-form');
  if (!form) return;

  const input = byId('command-input');
  const spinner = byId('cmd-spinner');
  const sendBtn = byId('cmd-send-btn');
  const responseEl = byId('last-response');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = (input?.value || '').trim();
    if (!text) return;

    if (spinner) spinner.classList.remove('hidden');
    if (sendBtn) sendBtn.disabled = true;

    // Show pending state in response
    if (responseEl) {
      responseEl.textContent = '';
    }

    try {
      const res = await fetch('/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      if (responseEl && data.response) {
        responseEl.textContent = data.response;
      }
      if (input) input.value = '';
    } catch (err) {
      if (responseEl) responseEl.textContent = 'Command failed — check connection.';
    } finally {
      if (spinner) spinner.classList.add('hidden');
      if (sendBtn) sendBtn.disabled = false;
    }
  });

  // Keyboard shortcut: Ctrl+K to focus command input
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
      e.preventDefault();
      input?.focus();
    }
  });
}


// ---------- Init ----------
document.addEventListener('DOMContentLoaded', () => {
  // Particles
  new ParticleCanvas('particle-canvas');

  // Sidebar
  initSidebar();

  // WebSocket (only on pages that have state display)
  if (byId('state-orb-container') || byId('stat-session')) {
    const wsDot = byId('ws-dot');
    const wsLabel = byId('ws-label');

    new JarvisWebSocket(
      (data) => renderState(data),
      (connected) => {
        if (wsDot) {
          wsDot.classList.toggle('connected', connected);
        }
        if (wsLabel) {
          wsLabel.textContent = connected ? 'Connected' : 'Reconnecting...';
        }
        if (!connected) {
          renderState({ state: 'OFFLINE' });
        }
      }
    );
  }

  // Command form
  initCommandForm();

  // Toast manager (global)
  window.toast = new ToastManager();
});
