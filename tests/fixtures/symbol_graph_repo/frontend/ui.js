// Tiny JS file to exercise cross-language tree-sitter parsing.
// SymbolIndex should still find renderApp / fetchUser as defs.

function renderApp(state) {
    const user = state.user;
    return `<div>${user.name}</div>`;
}

function fetchUser(id) {
    return fetch(`/api/users/${id}`).then(r => r.json());
}
