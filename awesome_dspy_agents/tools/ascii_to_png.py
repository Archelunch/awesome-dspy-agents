"""
ASCII to PNG Converter

This module provides functionality to convert ASCII art text into PNG images.
It uses PIL (Pillow) to render text with monospace fonts and save as image files.
"""

# pyright: reportMissingTypeStubs=false

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont  # type: ignore


class AsciiToPngConverter:
    """
    A class to convert ASCII art text into PNG images.

    This converter handles proper spacing, font selection, and image sizing
    to maintain the visual structure of ASCII art when converted to images.
    """

    def __init__(
        self,
        font_size: int = 16,
        background_color: tuple[int, int, int] = (255, 255, 255),
        text_color: tuple[int, int, int] = (0, 0, 0),
        padding: int = 20,
        char_width: int | None = None,
        char_height: int | None = None,
    ):
        """
        Initialize the ASCII to PNG converter.

        Args:
            font_size: Size of the font to use for rendering
            background_color: RGB tuple for background color (default: white)
            text_color: RGB tuple for text color (default: black)
            padding: Padding around the text in pixels
            char_width: Fixed character width in pixels (auto-calculated if None)
            char_height: Fixed character height in pixels (auto-calculated if None)
        """
        self.font_size = font_size
        self.background_color = background_color
        self.text_color = text_color
        self.padding = padding
        # Initialize then compute concrete int dimensions
        self.char_width: int = 0
        self.char_height: int = 0
        self.font = self._load_font()
        self._calculate_char_dimensions()

    def _load_font(self) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
        """
        Load a monospace font for proper ASCII art rendering.

        Returns:
            ImageFont object for text rendering
        """
        # Try to load common monospace fonts
        font_paths = [
            # macOS fonts
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/Monaco.ttf",
            "/Library/Fonts/Monaco.ttf",
            # Linux fonts
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            # Windows fonts
            "C:/Windows/Fonts/consola.ttf",
            "C:/Windows/Fonts/cour.ttf",
        ]

        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, self.font_size)
                except OSError:
                    continue

        # Fallback to default font
        try:
            return ImageFont.load_default()
        except OSError:
            # If all else fails, use PIL's basic font
            return ImageFont.load_default()

    def _calculate_char_dimensions(self):
        """
        Calculate character dimensions for consistent spacing.
        """
        # Use 'M' as it's typically the widest character in monospace fonts
        bbox = self.font.getbbox("M")
        self.char_width = int(bbox[2] - bbox[0])
        self.char_height = int(bbox[3] - bbox[1])

    def _calculate_text_dimensions(self, text: str) -> tuple[int, int]:
        """
        Calculate the dimensions needed to render the given text.

        Args:
            text: The ASCII text to measure

        Returns:
            Tuple of (width, height) in pixels
        """
        lines = text.split("\n")

        # Calculate width based on the longest line using fixed character width
        max_chars = max(len(line) for line in lines) if lines else 0
        max_width = max_chars * self.char_width

        # Calculate height using fixed character height
        total_height = len(lines) * self.char_height

        return max_width, total_height

    def convert_text_to_png(
        self,
        ascii_text: str,
        output_path: str,
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> str:
        """
        Convert ASCII text to a PNG image.

        Args:
            ascii_text: The ASCII art text to convert
            output_path: Path where the PNG file should be saved
            image_width: Optional fixed width for the image
            image_height: Optional fixed height for the image

        Returns:
            Path to the created PNG file
        """
        # Remove line numbers if present (format: "     1|content")
        cleaned_lines = []
        for line in ascii_text.split("\n"):
            # Check if line starts with line number format
            if "|" in line and line[: line.find("|")].strip().isdigit():
                # Extract content after the pipe
                cleaned_lines.append(line[line.find("|") + 1 :])
            else:
                cleaned_lines.append(line)

        cleaned_text = "\n".join(cleaned_lines)

        # Calculate image dimensions
        text_width, text_height = self._calculate_text_dimensions(cleaned_text)

        # Use provided dimensions or calculate from text
        img_width = image_width or (text_width + 2 * self.padding)
        img_height = image_height or (text_height + 2 * self.padding)

        # Create image with higher quality settings
        image = Image.new("RGB", (img_width, img_height), self.background_color)
        draw = ImageDraw.Draw(image)

        # Draw text character by character for precise positioning
        lines = cleaned_text.split("\n")

        for line_idx, line in enumerate(lines):
            y_pos = self.padding + (line_idx * self.char_height)

            for char_idx, char in enumerate(line):
                if char != " ":  # Skip spaces for efficiency
                    x_pos = self.padding + (char_idx * self.char_width)
                    draw.text(
                        (x_pos, y_pos), char, font=self.font, fill=self.text_color
                    )

        # Save the image
        image.save(output_path, "PNG")
        return output_path

    def convert_file_to_png(
        self,
        input_file: str | Path,
        output_file: str | Path | None = None,
    ) -> str:
        """
        Convert an ASCII art file to a PNG image.

        Args:
            input_file: Path to the ASCII art text file
            output_file: Optional output path for PNG file. If not provided,
                        will use the input filename with .png extension

        Returns:
            Path to the created PNG file
        """
        input_path = Path(input_file)

        # Read the ASCII text
        with open(input_path, encoding="utf-8") as f:
            ascii_text = f.read()

        # Determine output path
        if output_file is None:
            output_file = input_path.with_suffix(".png")

        return self.convert_text_to_png(ascii_text, str(output_file))

    def batch_convert_directory(
        self,
        input_dir: str | Path,
        output_dir: str | Path | None = None,
        file_pattern: str = "*.txt",
    ) -> list:
        """
        Convert all ASCII art files in a directory to PNG images.

        Args:
            input_dir: Directory containing ASCII art files
            output_dir: Directory to save PNG files. If None, saves in input_dir
            file_pattern: Glob pattern for input files (default: "*.txt")

        Returns:
            List of paths to created PNG files
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir) if output_dir else input_path

        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)

        created_files = []

        # Process all matching files
        for input_file in input_path.glob(file_pattern):
            output_file = output_path / f"{input_file.stem}.png"
            try:
                result = self.convert_file_to_png(input_file, output_file)
                created_files.append(result)
                print(f"Converted: {input_file.name} -> {output_file.name}")
            except Exception as e:
                print(f"Error converting {input_file.name}: {e}")

        return created_files

    def convert_text_to_png_hq(
        self,
        ascii_text: str,
        output_path: str,
        scale_factor: int = 2,
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> str:
        """
        Convert ASCII text to a high-quality PNG image with scaling.

        Args:
            ascii_text: The ASCII art text to convert
            output_path: Path where the PNG file should be saved
            scale_factor: Scaling factor for higher resolution (default: 2)
            image_width: Optional fixed width for the image
            image_height: Optional fixed height for the image

        Returns:
            Path to the created PNG file
        """
        # Create a temporary converter with scaled font size
        hq_font_size = self.font_size * scale_factor
        hq_padding = self.padding * scale_factor

        # Create temporary high-res converter
        temp_converter = AsciiToPngConverter(
            font_size=hq_font_size,
            background_color=self.background_color,
            text_color=self.text_color,
            padding=hq_padding,
        )

        # Create temporary high-res image
        temp_path = output_path.replace(".png", "_temp_hq.png")
        temp_converter.convert_text_to_png(
            ascii_text,
            temp_path,
            image_width and image_width * scale_factor,
            image_height and image_height * scale_factor,
        )

        # Load and resize with high-quality resampling
        high_res_image = Image.open(temp_path)
        if scale_factor > 1:
            # Calculate final size
            final_width = high_res_image.width // scale_factor
            final_height = high_res_image.height // scale_factor

            # Resize with high-quality filter
            final_image = high_res_image.resize(
                (final_width, final_height), Image.Resampling.LANCZOS
            )
        else:
            final_image = high_res_image

        # Save final image
        final_image.save(output_path, "PNG")

        # Clean up temporary file
        os.remove(temp_path)

        return output_path

    def convert_text_to_image_hq(
        self,
        ascii_text: str,
        scale_factor: int = 3,
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> Image.Image:
        """
        Convert ASCII text to a high-quality PIL Image in memory without saving to disk.

        Args:
            ascii_text: The ASCII art text to convert
            scale_factor: Scaling factor for higher resolution (default: 3)
            image_width: Optional fixed width for the image
            image_height: Optional fixed height for the image

        Returns:
            PIL Image object with high-quality rendering
        """
        # Create a high-resolution converter
        hq_font_size = self.font_size * scale_factor
        hq_padding = self.padding * scale_factor

        # Create temporary high-res converter
        temp_converter = AsciiToPngConverter(
            font_size=hq_font_size,
            background_color=self.background_color,
            text_color=self.text_color,
            padding=hq_padding,
        )

        # Remove line numbers if present (format: "     1|content")
        cleaned_lines = []
        for line in ascii_text.split("\n"):
            # Check if line starts with line number format
            if "|" in line and line[: line.find("|")].strip().isdigit():
                # Extract content after the pipe
                cleaned_lines.append(line[line.find("|") + 1 :])
            else:
                cleaned_lines.append(line)

        cleaned_text = "\n".join(cleaned_lines)

        # Calculate image dimensions at high resolution
        text_width, text_height = temp_converter._calculate_text_dimensions(
            cleaned_text
        )

        # Use provided dimensions or calculate from text
        img_width = (
            image_width * scale_factor if image_width else (text_width + 2 * hq_padding)
        )
        img_height = (
            image_height * scale_factor
            if image_height
            else (text_height + 2 * hq_padding)
        )

        # Create high-resolution image
        high_res_image = Image.new(
            "RGB", (img_width, img_height), self.background_color
        )
        draw = ImageDraw.Draw(high_res_image)

        # Draw text character by character for precise positioning
        lines = cleaned_text.split("\n")

        for line_idx, line in enumerate(lines):
            y_pos = hq_padding + (line_idx * temp_converter.char_height)

            for char_idx, char in enumerate(line):
                if char != " ":  # Skip spaces for efficiency
                    x_pos = hq_padding + (char_idx * temp_converter.char_width)
                    draw.text(
                        (x_pos, y_pos),
                        char,
                        font=temp_converter.font,
                        fill=self.text_color,
                    )

        # If scale factor > 1, resize with high-quality filter
        if scale_factor > 1:
            final_width = high_res_image.width // scale_factor
            final_height = high_res_image.height // scale_factor

            # Resize with high-quality filter
            final_image = high_res_image.resize(
                (final_width, final_height), Image.Resampling.LANCZOS
            )
        else:
            final_image = high_res_image

        return final_image


def create_png_from_ascii(
    ascii_text: str,
    output_path: str,
    font_size: int = 16,
    background_color: tuple[int, int, int] = (255, 255, 255),
    text_color: tuple[int, int, int] = (0, 0, 0),
    high_quality: bool = False,
    scale_factor: int = 2,
) -> str:
    """
    Convenience function to convert ASCII text to PNG with default settings.

    Args:
        ascii_text: The ASCII art text to convert
        output_path: Path where the PNG file should be saved
        font_size: Size of the font to use
        background_color: RGB tuple for background color
        text_color: RGB tuple for text color
        high_quality: Use high-quality rendering with scaling
        scale_factor: Scaling factor for high-quality mode

    Returns:
        Path to the created PNG file
    """
    converter = AsciiToPngConverter(
        font_size=font_size, background_color=background_color, text_color=text_color
    )

    if high_quality:
        return converter.convert_text_to_png_hq(ascii_text, output_path, scale_factor)
    else:
        return converter.convert_text_to_png(ascii_text, output_path)
