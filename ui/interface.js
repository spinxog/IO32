// ui/interface.js

document.getElementById("upload-form").addEventListener("submit", async function (e) {
  e.preventDefault();

  const form = new FormData();
  const imageInput = document.getElementById("image-input");
  form.append("image", imageInput.files[0]);

  const response = await fetch("/analyze", {
    method: "POST",
    body: form,
  });

  const data = await response.json();

  const feedbackEl = document.getElementById("feedback-text");
  const imageEl = document.getElementById("preview-image");

  if (data.error) {
    feedbackEl.textContent = "Error: " + data.error;
    imageEl.style.display = "none";
  } else {
    feedbackEl.textContent = data.feedback;
    imageEl.src = data.image_url;
    imageEl.style.display = "block";
  }
});
