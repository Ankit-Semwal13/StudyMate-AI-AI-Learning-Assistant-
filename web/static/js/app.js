(function () {
  "use strict";

  // ---------------- Theme toggle (sidebar Dark Mode switch) ----------------
  var root = document.documentElement;
  var darkSwitch = document.getElementById("darkmode-switch");
  var savedTheme = localStorage.getItem("studymate-theme");
  if (savedTheme) root.setAttribute("data-theme", savedTheme);

  function syncSwitch() {
    if (!darkSwitch) return;
    darkSwitch.checked = root.getAttribute("data-theme") !== "light";
  }
  syncSwitch();

  if (darkSwitch) {
    darkSwitch.addEventListener("change", function () {
      var next = darkSwitch.checked ? "dark" : "light";
      root.setAttribute("data-theme", next);
      localStorage.setItem("studymate-theme", next);
    });
  }

  // ---------------- Upload modal ----------------
  var backdrop = document.getElementById("upload-modal-backdrop");
  var closeBtn = document.getElementById("upload-modal-close");
  var tabs = document.querySelectorAll(".modal-tab");
  var urlInput = document.getElementById("url-input");
  var fileInput = document.getElementById("file-input");
  var startBtn = document.getElementById("start-processing-btn");
  var progressWrap = document.getElementById("job-progress");
  var progressFill = document.getElementById("job-progress-fill");
  var progressLabel = document.getElementById("job-progress-label");
  var jobError = document.getElementById("job-error");

  function openModal(tab) {
    if (!backdrop) return;
    backdrop.classList.add("open");
    if (tab) setActiveTab(tab);
  }
  function closeModal() {
    if (!backdrop) return;
    backdrop.classList.remove("open");
  }
  function setActiveTab(name) {
    tabs.forEach(function (t) {
      var active = t.getAttribute("data-tab") === name;
      t.classList.toggle("active", active);
    });
    document.querySelectorAll(".modal-body").forEach(function (p) {
      p.style.display = p.getAttribute("data-panel") === name ? "" : "none";
    });
  }

  document.querySelectorAll("[data-open-upload]").forEach(function (el) {
    el.addEventListener("click", function (e) {
      e.preventDefault();
      openModal(el.getAttribute("data-open-upload-tab"));
    });
  });
  if (closeBtn) closeBtn.addEventListener("click", closeModal);
  if (backdrop) {
    backdrop.addEventListener("click", function (e) {
      if (e.target === backdrop) closeModal();
    });
  }
  tabs.forEach(function (t) {
    t.addEventListener("click", function () {
      setActiveTab(t.getAttribute("data-tab"));
    });
  });

  var polling = null;

  function pollJob(jobId) {
    if (polling) clearInterval(polling);
    polling = setInterval(function () {
      fetch("/api/jobs/" + jobId)
        .then(function (r) { return r.json(); })
        .then(function (job) {
          var pct = Math.round((job.fraction || 0) * 100);
          if (progressFill) progressFill.style.width = pct + "%";
          if (progressLabel) progressLabel.textContent = job.stage || "Working...";

          if (job.status === "done") {
            clearInterval(polling);
            if (progressLabel) progressLabel.textContent = "Done! Redirecting...";
            setTimeout(function () {
              window.location.href = "/notes/" + job.slug;
            }, 600);
          } else if (job.status === "error") {
            clearInterval(polling);
            if (jobError) {
              jobError.style.display = "block";
              jobError.textContent = job.error || "Something went wrong.";
            }
            if (startBtn) { startBtn.disabled = false; startBtn.textContent = "🚀 Process Video"; }
          }
        })
        .catch(function () { /* transient network hiccup, keep polling */ });
    }, 1500);
  }

  if (startBtn) {
    startBtn.addEventListener("click", function () {
      var activeTab = document.querySelector(".modal-tab.active").getAttribute("data-tab");
      var formData = new FormData();

      if (activeTab === "url") {
        if (!urlInput.value.trim()) { urlInput.focus(); return; }
        formData.append("url", urlInput.value.trim());
      } else {
        if (!fileInput.files.length) { fileInput.focus(); return; }
        formData.append("file", fileInput.files[0]);
      }

      if (jobError) jobError.style.display = "none";
      startBtn.disabled = true;
      startBtn.textContent = "Processing...";
      if (progressWrap) progressWrap.style.display = "block";
      if (progressFill) progressFill.style.width = "0%";
      if (progressLabel) progressLabel.textContent = "Uploading...";

      fetch("/api/process", { method: "POST", body: formData })
        .then(function (r) {
          if (!r.ok) return r.json().then(function (e) { throw new Error(e.detail || "Failed to start"); });
          return r.json();
        })
        .then(function (data) { pollJob(data.job_id); })
        .catch(function (err) {
          startBtn.disabled = false;
          startBtn.textContent = "🚀 Process Video";
          if (jobError) {
            jobError.style.display = "block";
            jobError.textContent = err.message || "Something went wrong.";
          }
        });
    });
  }

  // Auto-open modal if URL has #upload
  if (window.location.hash === "#upload") openModal();
})();
