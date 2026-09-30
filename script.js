document.addEventListener("DOMContentLoaded", function () {

    const form = document.querySelector("form");
    const button = form ? form.querySelector("button[type='submit']") : null;

    const fileInput = document.getElementById("resume");
    const fileName = document.getElementById("file-name");

    if (fileInput && fileName) {
        fileInput.addEventListener("change", function () {

            if (this.files.length > 0) {
                fileName.textContent =
                    "Selected: " + this.files[0].name;
            } else {
                fileName.textContent =
                    "No file selected";
            }

        });
    }

    if (form) {

        form.addEventListener("submit", function () {

            if (button) {
                button.disabled = true;
                button.innerHTML = "⏳ Analyzing Resume...";
            }

        });

    }

});