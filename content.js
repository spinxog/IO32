(function () {
  const btn = document.createElement("button");
  btn.textContent = "Analyze Design";
  btn.style.cssText = "position:fixed;top:10px;right:10px;z-index:9999;padding:8px;";
  document.body.appendChild(btn);

  btn.onclick = captureCanvasAndSend;

  function captureCanvasAndSend() {
    const canvas = document.querySelector("canvas");
    if (!canvas) return alert("Canvas not found!");

    canvas.toBlob(async (blob) => {
      const formData = new FormData();
      formData.append("image", blob, "design.png");

      try {
        const response = await fetch("http://192.168.1.65:5000/analyze", {
          method: "POST",
          body: formData,
        });

        const data = await response.json();
        alert("Feedback:\n" + data.feedback);
      } catch (e) {
        alert("Failed to send image: " + e.message);
      }
    }, "image/png");
  }
})();
