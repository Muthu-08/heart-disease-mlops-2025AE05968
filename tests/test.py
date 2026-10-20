import io
import os
import sys
import pytest
from PIL import Image, ImageDraw, ImageFont

# Ensure the screenshots directory exists
os.makedirs("screenshots", exist_ok=True)

# 1. Capture Pytest console output
stdout_capture = io.StringIO()
old_stdout = sys.stdout
sys.stdout = stdout_capture

# Run pytest on the tests directory
exit_code = pytest.main(["tests/", "-v", "--color=no"])

# Restore stdout
sys.stdout = old_stdout
test_output = stdout_capture.getvalue()

# Print output to terminal as well
print(test_output)

# 2. Render captured terminal text into an image
lines = test_output.strip().split("\n")

# Use a default monospace font
try:
    font = ImageFont.truetype("consola.ttf", 15)  # Windows Consolas
except IOError:
    font = ImageFont.load_default()

# Calculate dynamic image dimensions based on text length
max_line_len = max(len(line) for line in lines) if lines else 60
char_width = 9
line_height = 22
padding = 25

img_width = max(750, max_line_len * char_width + 2 * padding)
img_height = len(lines) * line_height + 2 * padding

# Create dark terminal-style canvas
img = Image.new("RGB", (img_width, img_height), color="#1e1e1e")
draw = ImageDraw.Draw(img)

# 3. Draw text with status-based syntax highlighting
y_text = padding
for line in lines:
    text_color = "#d4d4d4"  # Default light grey

    if "PASSED" in line:
        text_color = "#4ec9b0"  # Terminal green
    elif "FAILED" in line or "ERROR" in line:
        text_color = "#f44747"  # Terminal red
    elif "===" in line or "rootdir:" in line:
        text_color = "#858585"  # Muted grey
    elif "collected" in line:
        text_color = "#569cd6"  # Light blue

    draw.text((padding, y_text), line, fill=text_color, font=font)
    y_text += line_height

# 4. Save screenshot
output_file = "screenshots/04_unit_tests_passing.png"
img.save(output_file)
print(f"\n[✓] Unit test evidence generated and saved to: {output_file}")