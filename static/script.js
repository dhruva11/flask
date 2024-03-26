function showSelectedImage(input) {
    const fileInput = document.getElementById('file');
    const selectedImage = document.getElementById('selected-image');

    if (fileInput.files.length > 0) {
        selectedImage.style.display = 'block';
        selectedImage.textContent = `Selected Image: ${fileInput.files[0].name}`;
    } else {
        selectedImage.style.display = 'none';
    }
}
