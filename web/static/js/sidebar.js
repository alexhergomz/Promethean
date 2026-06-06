/* Sidebar: session list, search filter, context menu (rename/delete/export),
 * new session, switch session. */

Object.assign(ChatApp.prototype, {

  async loadSessions() {
    try {
      const r = await this._fetchAuth('/api/sessions');
      const data = await r.json();
      this._sessions = data.sessions || [];
      this._renderSessionList();
    } catch(e) { console.error('loadSessions:', e); }
  },

  _renderSessionList() {
    const list = document.getElementById('session-list');
    if (!list) return;
    const q = (document.getElementById('sess-search-input')?.value || '')
      .trim().toLowerCase();
    const items = (this._sessions || []).filter(s =>
      !q || (s.title || '').toLowerCase().includes(q) || s.id.includes(q)
    );
    list.innerHTML = '';
    if (items.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'sess-empty';
      empty.textContent = q
        ? 'No sessions match.' : 'No sessions yet — click + New.';
      list.appendChild(empty);
      return;
    }
    items.forEach(s => {
      const el = document.createElement('div');
      el.className = 'sess-item' + (s.id === this.sessionId ? ' active' : '');
      const title = s.title && s.title !== 'New chat'
        ? s.title : `Untitled (${s.id.slice(0, 6)})`;
      el.innerHTML = `
        <div class="sess-title">
          <span class="sess-dot ${s.busy ? '' : 'idle'}"></span>
          <span>${this._escapeHtml(title)}</span>
        </div>
        <div class="sess-info">
          <span>${s.message_count || 0} msg</span>
          <span>${this._fmtRelTime(s.last_active)}</span>
        </div>`;
      el.onclick = () => this.switchSession(s.id);
      el.oncontextmenu = (e) => {
        e.preventDefault();
        this._showSessMenu(e.clientX, e.clientY, s);
      };
      list.appendChild(el);
    });
  },

  _showSessMenu(x, y, sess) {
    const menu = document.getElementById('sess-menu');
    menu.innerHTML = `
      <div class="menu-item" data-act="rename">Rename...</div>
      <div class="menu-item" data-act="export">Export Markdown</div>
      <div class="menu-sep"></div>
      <div class="menu-item danger" data-act="delete">Delete</div>`;
    menu.querySelectorAll('.menu-item').forEach(item => {
      item.onclick = () => {
        menu.style.display = 'none';
        const act = item.dataset.act;
        if (act === 'rename') this.renameSession(sess);
        else if (act === 'export') this.exportSession(sess);
        else if (act === 'delete') this.deleteSession(sess);
      };
    });
    menu.style.display = 'block';
    const rect = menu.getBoundingClientRect();
    const px = Math.min(x, window.innerWidth - rect.width - 8);
    const py = Math.min(y, window.innerHeight - rect.height - 8);
    menu.style.left = px + 'px';
    menu.style.top = py + 'px';
    const dismiss = (ev) => {
      if (!menu.contains(ev.target)) {
        menu.style.display = 'none';
        document.removeEventListener('click', dismiss);
      }
    };
    setTimeout(() => document.addEventListener('click', dismiss), 0);
  },

  async renameSession(sess) {
    const title = prompt('Rename session:', sess.title || '');
    if (title === null) return;
    const t = title.trim();
    if (!t) return;
    try {
      const r = await this._fetchAuth(`/api/sessions/${sess.id}`, {
        method: 'PATCH',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({title: t}),
      });
      if (r.ok) this.loadSessions();
    } catch(e) { console.error('rename:', e); }
  },

  async deleteSession(sess) {
    if (!confirm(`Delete session "${sess.title || sess.id}"?\n\n` +
                 `This removes ${sess.message_count || 0} messages permanently.`)) {
      return;
    }
    try {
      const r = await this._fetchAuth(`/api/sessions/${sess.id}`, {
        method: 'DELETE',
      });
      if (r.ok) {
        if (this.sessionId === sess.id) {
          this._disconnectWS();
          this.sessionId = null;
          this._clearChat();
          this._showWelcome();
        }
        this.loadSessions();
      }
    } catch(e) { console.error('delete:', e); }
  },

  exportSession(sess) {
    window.location.href = `/api/sessions/${sess.id}/export`;
  },

  async newSession() {
    this._disconnectWS();
    this._clearChat();
    try {
      const r = await this._fetchAuth('/api/prompt', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({prompt: '', session_id: ''}),
      });
      const data = await r.json();
      if (r.ok && data.session_id) {
        this.sessionId = data.session_id;
        this._connectWS(this.sessionId);
        this._showWelcome();
      }
    } catch(e) { console.error('newSession:', e); }
    this.loadSessions();
  },

  async switchSession(sid) {
    if (sid === this.sessionId) return;
    this._disconnectWS();
    this.sessionId = sid;
    this._clearChat();
    try {
      const r = await this._fetchAuth(`/api/sessions/${sid}`);
      const data = await r.json();
      (data.messages || []).forEach(m => {
        if (m.role === 'user') this._addUserBubble(m.content);
        else if (m.role === 'assistant') {
          this._addAssistantBubble(m.content);
          if (m.tool_calls) m.tool_calls.forEach(tc => {
            this._addToolCard(tc.name, tc.inputs, tc.status, tc.result);
          });
        }
      });
    } catch(e) { console.error('switchSession:', e); }
    this._connectWS(sid);
    this.loadSessions();
  },
});
