/* Bootstrap: instantiate ChatApp once all mixin modules have loaded,
 * then wire keyboard + sidebar click-outside handlers. */

const app = new ChatApp();
app.initTheme();
app.bootstrap();
app._showWelcome();

const promptInput = document.getElementById('prompt-input');
promptInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    app.send();
  }
});
promptInput.addEventListener('input', () => {
  promptInput.style.height = 'auto';
  promptInput.style.height = Math.min(promptInput.scrollHeight, 200) + 'px';
});

document.getElementById('main').addEventListener('click', () => {
  document.getElementById('sidebar').classList.remove('open');
});
