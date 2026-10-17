const topics = {
  finance: {
    panelLabel: "finance-tab",
    kicker: "FEE SUPPORT",
    title: "What would you like to know?",
    placeholder: "For example: What is the tuition fee for DEMO-CBSE Class 1 in 2026-2027?",
    helper: "Include the board, class, and academic year for fee questions.",
    suggestions: [
      "Tuition fee for DEMO-CBSE Class 1 in 2026-2027",
      "What fees are charged annually?",
      "Fee for DEMO-CBSE Class 5 in 2026-2027",
    ],
  },
  academic: {
    panelLabel: "academic-tab",
    kicker: "ACADEMIC SUPPORT",
    title: "What are you learning about?",
    placeholder: "For example: Which books are recommended for Standard 1 English?",
    helper: "Ask about subjects, syllabus, exams, or recommended books.",
    suggestions: [
      "Which subjects are taught in Standard 1?",
      "Which books are recommended for Standard 1?",
      "Recommended books for Standard 1 English",
      "What does a Standard 5 student learn in science?",
    ],
  },
};

const tabButtons = [...document.querySelectorAll(".tab-button")];
const panel = document.querySelector("#query-panel");
const form = document.querySelector("#query-form");
const input = document.querySelector("#query-input");
const sendButton = document.querySelector("#send-button");
const sendLabel = document.querySelector("#send-label");
const responseCard = document.querySelector("#response-card");
const responseAnswer = document.querySelector("#response-answer");
const intentBadge = document.querySelector("#intent-badge");
const limitations = document.querySelector("#limitations");
const requestReference = document.querySelector("#request-reference");
const clarificationForm = document.querySelector("#clarification-form");
const clarificationInput = document.querySelector("#clarification-input");
const clarificationSubmit = document.querySelector("#clarification-submit");
const clarificationLabel = document.querySelector("#clarification-label");
const clarificationOptions = document.querySelector("#clarification-options");

let activeTopic = "finance";
let pendingRequestId = null;
let inputRequestKind = null;
let visibleMcpElicitationId = null;

function renderSuggestions(topic) {
  const container = document.querySelector("#suggestions");
  container.replaceChildren();
  for (const suggestionText of topics[topic].suggestions) {
    const button = document.createElement("button");
    button.className = "suggestion";
    button.type = "button";
    button.textContent = suggestionText;
    button.addEventListener("click", () => {
      input.value = suggestionText;
      input.focus();
    });
    container.append(button);
  }
}

function selectTopic(topic) {
  if (topic !== activeTopic) {
    input.value = "";
  }
  activeTopic = topic;
  const config = topics[topic];
  for (const button of tabButtons) {
    const selected = button.dataset.topic === topic;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
  }
  panel.setAttribute("aria-labelledby", config.panelLabel);
  document.querySelector("#topic-kicker").textContent = config.kicker;
  document.querySelector("#topic-title").textContent = config.title;
  document.querySelector("#helper-text").textContent = config.helper;
  input.placeholder = config.placeholder;
  responseCard.hidden = true;
  renderSuggestions(topic);
}

for (const button of tabButtons) {
  button.addEventListener("click", () => selectTopic(button.dataset.topic));
  button.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const nextTopic = activeTopic === "finance" ? "academic" : "finance";
    document.querySelector(`[data-topic="${nextTopic}"]`).focus();
    selectTopic(nextTopic);
  });
}

function showResponse(data) {
  pendingRequestId = null;
  inputRequestKind = null;
  visibleMcpElicitationId = null;
  clarificationForm.hidden = true;
  responseAnswer.textContent = data.answer || "The assistant returned an empty response.";
  intentBadge.textContent = data.intent || activeTopic;
  intentBadge.hidden = !data.intent;
  limitations.replaceChildren();
  const notes = Array.isArray(data.limitations) ? data.limitations : [];
  limitations.hidden = notes.length === 0;
  for (const note of notes) {
    const item = document.createElement("li");
    item.textContent = note;
    limitations.append(item);
  }
  requestReference.textContent = data.request_id ? `Request ID: ${data.request_id}` : "";
  responseCard.hidden = false;
  responseCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function showClarification(data) {
  pendingRequestId = data.request_id;
  inputRequestKind = "langgraph";
  responseAnswer.textContent = data.question || "Please clarify your request.";
  intentBadge.textContent = data.intent || "clarification";
  intentBadge.hidden = false;
  limitations.replaceChildren();
  limitations.hidden = true;
  requestReference.textContent = data.request_id ? `Request ID: ${data.request_id}` : "";
  clarificationInput.value = "";
  clarificationOptions.replaceChildren();
  const options = Array.isArray(data.options) ? data.options : [];
  for (const option of options) {
    const button = document.createElement("button");
    button.className = "clarification-option";
    button.type = "button";
    button.textContent = option;
    button.addEventListener("click", () => {
      clarificationInput.value = option;
      clarificationForm.requestSubmit();
    });
    clarificationOptions.append(button);
  }
  if (data.kind === "fee_board_selection") {
    const cancel = document.createElement("button");
    cancel.className = "clarification-option";
    cancel.type = "button";
    cancel.textContent = "Cancel lookup";
    cancel.addEventListener("click", () => {
      clarificationInput.value = "no";
      clarificationForm.requestSubmit();
    });
    clarificationOptions.append(cancel);
  }
  clarificationForm.hidden = false;
  responseCard.hidden = false;
  responseCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  clarificationInput.focus();
}

function showMcpElicitation(data) {
  pendingRequestId = data.request_id;
  inputRequestKind = "mcp";
  visibleMcpElicitationId = data.request_id;
  responseAnswer.textContent = data.question || "Please choose a language for the book search.";
  intentBadge.textContent = "academic";
  intentBadge.hidden = false;
  limitations.replaceChildren();
  limitations.hidden = true;
  requestReference.textContent = `Request ID: ${data.request_id}`;
  clarificationInput.value = "";
  clarificationOptions.replaceChildren();
  for (const option of (Array.isArray(data.options) ? data.options : [])) {
    const button = document.createElement("button");
    button.className = "clarification-option";
    button.type = "button";
    button.textContent = option;
    button.addEventListener("click", () => {
      clarificationInput.value = option;
      clarificationForm.requestSubmit();
    });
    clarificationOptions.append(button);
  }
  const cancel = document.createElement("button");
  cancel.className = "clarification-option";
  cancel.type = "button";
  cancel.textContent = "Cancel question";
  cancel.addEventListener("click", () => {
    clarificationInput.value = "cancel";
    clarificationForm.requestSubmit();
  });
  clarificationOptions.append(cancel);
  clarificationForm.hidden = false;
  responseCard.hidden = false;
  responseCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  clarificationInput.focus();
}

function showQueryResult(data) {
  if (data.status === "needs_input") showClarification(data);
  else showResponse(data);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = input.value.trim();
  if (!query) {
    input.focus();
    return;
  }

  sendButton.disabled = true;
  sendLabel.textContent = "Working…";
  pendingRequestId = null;
  inputRequestKind = null;
  visibleMcpElicitationId = null;
  clarificationForm.hidden = true;
  responseCard.hidden = true;
  try {
    const requestId = crypto.randomUUID();
    const queryPromise = fetch("/api/v1/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, request_id: requestId }),
    }).then(async (response) => ({ response, data: await response.json() }))
      .catch((error) => ({ error }));

    let queryResult = null;
    while (queryResult === null) {
      queryResult = await Promise.race([
        queryPromise,
        new Promise((resolve) => window.setTimeout(() => resolve(null), 400)),
      ]);
      if (queryResult !== null) break;

      try {
        const elicitationResponse = await fetch(
          `/api/v1/elicitation/${encodeURIComponent(requestId)}`,
        );
        const elicitation = await elicitationResponse.json();
        if (elicitation.status === "pending" && visibleMcpElicitationId !== requestId) {
          showMcpElicitation(elicitation);
        }
      } catch {
        // Continue waiting for the original query response; the API reports failures there.
      }
    }

    if (queryResult.error) throw queryResult.error;
    const { response, data } = queryResult;
    if (!response.ok) {
      throw new Error(data.error || "The request could not be processed.");
    }
    showQueryResult(data);
  } catch (error) {
    showResponse({
      answer: error instanceof Error ? error.message : "Could not reach the support service.",
      intent: "error",
    });
  } finally {
    sendButton.disabled = false;
    sendLabel.textContent = "Ask question";
  }
});

clarificationForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const answer = clarificationInput.value.trim();
  if (!answer || !pendingRequestId) {
    clarificationInput.focus();
    return;
  }

  clarificationSubmit.disabled = true;
  clarificationLabel.textContent = inputRequestKind === "mcp" ? "Sending…" : "Continuing…";
  try {
    if (inputRequestKind === "mcp") {
      const response = await fetch(
        `/api/v1/elicitation/${encodeURIComponent(pendingRequestId)}/answer`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ answer }),
        },
      );
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "The answer could not be submitted.");
      responseAnswer.textContent = "Thanks. Searching the book recommendations now…";
      clarificationForm.hidden = true;
      inputRequestKind = null;
      visibleMcpElicitationId = null;
      return;
    }

    const response = await fetch(
      `/api/v1/query/${encodeURIComponent(pendingRequestId)}/resume`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ answer }),
      },
    );
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "The clarification could not be submitted.");
    }
    showQueryResult(data);
  } catch (error) {
    responseAnswer.textContent = error instanceof Error
      ? error.message
      : "Could not submit your clarification.";
    responseCard.hidden = false;
  } finally {
    clarificationSubmit.disabled = false;
    clarificationLabel.textContent = "Send answer";
  }
});

renderSuggestions(activeTopic);
