// Sign-in with the Cognito hosted login page (authorization code flow with PKCE).
// Defines window.authFetch: fetch() that sends the access token and re-logs-in on 401.
// When the server reports auth disabled (local development) it behaves like plain fetch.
(() => {
  const TOKEN_KEY = "support.access_token";
  const VERIFIER_KEY = "support.pkce_verifier";
  const STATE_KEY = "support.oauth_state";
  let config = { enabled: false };

  const read = (key) => {
    try { return sessionStorage.getItem(key); } catch { return null; }
  };
  const write = (key, value) => {
    try {
      if (value === null) sessionStorage.removeItem(key);
      else sessionStorage.setItem(key, value);
    } catch { /* storage unavailable: the user will simply sign in again */ }
  };

  const base64Url = (bytes) =>
    btoa(String.fromCharCode(...new Uint8Array(bytes)))
      .replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  const randomString = () => base64Url(crypto.getRandomValues(new Uint8Array(32)));
  const redirectUri = () => `${window.location.origin}/`;
  const authBase = () => {
    const domain = config.cognito_domain;
    return /^https?:\/\//.test(domain) ? domain : `https://${domain}`;
  };

  async function startLogin() {
    const verifier = randomString();
    const state = randomString();
    write(VERIFIER_KEY, verifier);
    write(STATE_KEY, state);
    const challenge = base64Url(
      await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier)),
    );
    const params = new URLSearchParams({
      response_type: "code",
      client_id: config.client_id,
      redirect_uri: redirectUri(),
      scope: "openid email",
      state,
      code_challenge: challenge,
      code_challenge_method: "S256",
    });
    window.location.assign(`${authBase()}/oauth2/authorize?${params}`);
    return new Promise(() => {}); // the page is navigating away
  }

  async function completeLogin(code, state) {
    const verifier = read(VERIFIER_KEY);
    const expectedState = read(STATE_KEY);
    write(VERIFIER_KEY, null);
    write(STATE_KEY, null);
    window.history.replaceState({}, "", window.location.pathname);
    if (!verifier || !state || state !== expectedState) return false;

    const response = await fetch(`${authBase()}/oauth2/token`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        grant_type: "authorization_code",
        client_id: config.client_id,
        code,
        redirect_uri: redirectUri(),
        code_verifier: verifier,
      }),
    });
    if (!response.ok) return false;
    write(TOKEN_KEY, (await response.json()).access_token);
    return true;
  }

  async function init() {
    try {
      config = await (await fetch("/api/v1/auth/config")).json();
    } catch {
      return; // the API itself will report problems on the first real request
    }
    if (!config.enabled) return;

    const params = new URLSearchParams(window.location.search);
    if (params.has("code") && !(await completeLogin(params.get("code"), params.get("state")))) {
      write(TOKEN_KEY, null);
    }
    if (!read(TOKEN_KEY)) await startLogin();
  }

  window.authReady = init();
  window.authFetch = async (url, options = {}) => {
    await window.authReady;
    if (!config.enabled) return fetch(url, options);
    const headers = { ...(options.headers || {}), Authorization: `Bearer ${read(TOKEN_KEY)}` };
    const response = await fetch(url, { ...options, headers });
    if (response.status === 401) {
      write(TOKEN_KEY, null);
      await startLogin();
    }
    return response;
  };
})();
