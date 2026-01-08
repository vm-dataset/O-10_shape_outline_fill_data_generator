"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        SHAPE FILL-TO-OUTLINE TASK GENERATOR                  ║
║                                                                               ║
║  Generates analog shape fill-to-outline tasks (A:B :: C:?)                   ║
║  Example: filled_circle → outline_circle :: filled_square → outline_square   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import random
import tempfile
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Any

from core import BaseGenerator, TaskPair, ImageRenderer
from core.video_utils import VideoGenerator
from .config import TaskConfig
from .prompts import get_prompt


class TaskGenerator(BaseGenerator):
    """
    Shape fill-to-outline task generator.
    
    Creates visual analogies in the format A:B :: C:?
    where shapes undergo fill-to-outline transformations.
    """
    
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.renderer = ImageRenderer(image_size=config.image_size)
        
        # Initialize video generator if enabled
        self.video_generator = None
        if config.generate_videos and VideoGenerator.is_available():
            self.video_generator = VideoGenerator(fps=config.video_fps, output_format="mp4")
        
        # Shape definitions - expanded set of shapes that work well with fill/outline
        self.base_shapes = [
            "square", "triangle", "circle", "diamond", "pentagon", "hexagon",
            "rectangle", "oval", "star", "heart", "cross", "arrow", "trapezoid",
            "rhombus", "octagon", "crescent", "plus", "minus", "L_shape", "T_shape"
        ]
        
        # Transformation types - multiple ways to transform fill/outline
        self.transformation_types = [
            "fill_to_outline",      # filled → outline
            "outline_to_fill",      # outline → filled
            "thick_to_thin",        # thick outline → thin outline
            "thin_to_thick",        # thin outline → thick outline
            "solid_to_dashed",      # solid outline → dashed outline
            "dashed_to_solid"       # dashed outline → solid outline
        ]
        
        # Fill styles - expanded with more variations
        self.fill_styles = {
            "filled": {"fill": True, "outline_width": 2, "dash": None},
            "outline": {"fill": False, "outline_width": 3, "dash": None},
            "thick_outline": {"fill": False, "outline_width": 5, "dash": None},
            "thin_outline": {"fill": False, "outline_width": 1, "dash": None},
            "dashed_outline": {"fill": False, "outline_width": 3, "dash": [5, 3]},
            "solid_outline": {"fill": False, "outline_width": 3, "dash": None}
        }
        
        # Base colors - multiple color options for variety
        self.base_colors = [
            (70, 130, 180),   # Steel Blue
            (220, 20, 60),    # Crimson
            (50, 205, 50),    # Lime Green
            (255, 140, 0),    # Dark Orange
            (128, 0, 128),    # Purple
            (255, 20, 147),   # Deep Pink
            (0, 128, 0),      # Dark Green
            (165, 42, 42),    # Brown
            (75, 0, 130),     # Indigo
            (255, 165, 0),    # Orange
        ]
        
        # Track generated combinations to prevent duplicates
        self.generated_combinations = set()
    
    def generate_task_pair(self, task_id: str) -> TaskPair:
        """Generate one shape fill-to-outline task pair."""
        
        # Generate task data
        task_data = self._generate_task_data()
        
        # Render images
        first_image = self._render_initial_state(task_data)
        final_image = self._render_final_state(task_data)
        
        # Generate video (optional)
        video_path = None
        if self.config.generate_videos and self.video_generator:
            video_path = self._generate_video(first_image, final_image, task_id, task_data)
        
        # Select prompt
        prompt = get_prompt(task_data.get("transformation_type", "default"))
        
        return TaskPair(
            task_id=task_id,
            domain=self.config.domain,
            prompt=prompt,
            first_image=first_image,
            final_image=final_image,
            ground_truth_video=video_path
        )
    
    # ══════════════════════════════════════════════════════════════════════════
    #  TASK DATA GENERATION
    # ══════════════════════════════════════════════════════════════════════════
    
    def _generate_task_data(self) -> Dict[str, Any]:
        """Generate transformation task data with duplicate prevention."""
        
        # Calculate total possible unique combinations
        num_shapes = len(self.base_shapes)
        num_transformations = len(self.transformation_types)
        num_colors = len(self.base_colors)
        max_unique_combinations = num_shapes * (num_shapes - 1) * num_transformations * num_colors
        
        # If we haven't exhausted all combinations, ensure uniqueness
        if len(self.generated_combinations) < max_unique_combinations:
            max_attempts = 1000  # Increase attempts for better coverage
            for attempt in range(max_attempts):
                # Select two different shapes for the analogy
                shape_a, shape_c = random.sample(self.base_shapes, 2)
                # Select transformation type
                transformation_type = random.choice(self.transformation_types)
                # Select color
                color = random.choice(self.base_colors)
                
                # Create a unique identifier for this combination
                combination_key = (shape_a, shape_c, transformation_type, color)
                
                # Check if this combination has been used before
                if combination_key not in self.generated_combinations:
                    self.generated_combinations.add(combination_key)
                    return self._generate_transformation_task(shape_a, shape_c, transformation_type, color)
            
            # If we still can't find a unique combination after many attempts,
            # generate all remaining combinations systematically
            return self._generate_systematic_unique_combination()
        
        # If we've exhausted unique combinations, allow duplicates but warn
        if len(self.generated_combinations) == max_unique_combinations:
            print(f"⚠️  Warning: Generated all {max_unique_combinations} unique combinations. Allowing duplicates for remaining tasks.")
        
        shape_a, shape_c = random.sample(self.base_shapes, 2)
        transformation_type = random.choice(self.transformation_types)
        color = random.choice(self.base_colors)
        return self._generate_transformation_task(shape_a, shape_c, transformation_type, color)
    
    def _generate_systematic_unique_combination(self) -> Dict[str, Any]:
        """Generate a unique combination systematically when random selection fails."""
        # Generate all possible combinations and find one not yet used
        for shape_a in self.base_shapes:
            for shape_c in self.base_shapes:
                if shape_a == shape_c:
                    continue
                for transformation_type in self.transformation_types:
                    for color in self.base_colors:
                        combination_key = (shape_a, shape_c, transformation_type, color)
                        if combination_key not in self.generated_combinations:
                            self.generated_combinations.add(combination_key)
                            return self._generate_transformation_task(shape_a, shape_c, transformation_type, color)
        
        # This should never happen if our math is correct
        raise RuntimeError("Failed to generate unique combination - this should not happen!")
    
    def _generate_transformation_task(self, shape_a: str, shape_c: str, transformation_type: str, color: tuple) -> Dict[str, Any]:
        """Generate a transformation task based on the transformation type."""
        
        # Map transformation types to style pairs
        style_mappings = {
            "fill_to_outline": ("filled", "outline"),
            "outline_to_fill": ("outline", "filled"),
            "thick_to_thin": ("thick_outline", "thin_outline"),
            "thin_to_thick": ("thin_outline", "thick_outline"),
            "solid_to_dashed": ("solid_outline", "dashed_outline"),
            "dashed_to_solid": ("dashed_outline", "solid_outline")
        }
        
        style_from, style_to = style_mappings[transformation_type]
        
        return {
            "transformation_type": transformation_type,
            "shape_a": shape_a,
            "shape_b": shape_a,  # Same shape, different style
            "shape_c": shape_c,
            "shape_d": shape_c,  # Same shape, different style
            "style_from": style_from,
            "style_to": style_to,
            "color": color,
            "description": f"{style_from} {shape_a} becomes {style_to} {shape_a}, {style_from} {shape_c} becomes {style_to} {shape_c}"
        }
    
    # ══════════════════════════════════════════════════════════════════════════
    #  IMAGE RENDERING
    # ══════════════════════════════════════════════════════════════════════════
    
    def _render_initial_state(self, task_data: Dict[str, Any]) -> Image.Image:
        """Render the initial state with A:B :: C:? layout."""
        img = self.renderer.create_blank_image()
        draw = ImageDraw.Draw(img)
        
        width, height = self.config.image_size
        margin = self.config.margin
        shape_size = self.config.shape_size
        
        # Layout positions
        # A    →    B
        # C    →    ?
        
        positions = {
            "A": (margin + shape_size//2, height//4),
            "arrow1": (width//2, height//4),
            "B": (width - margin - shape_size//2, height//4),
            "C": (margin + shape_size//2, 3*height//4),
            "arrow2": (width//2, 3*height//4),
            "question": (width - margin - shape_size//2, 3*height//4)
        }
        
        # Draw shapes and arrows using task data styles and colors
        color = task_data["color"]
        style_from = task_data["style_from"]
        style_to = task_data["style_to"]
        
        self._draw_shape_at_position(draw, task_data["shape_a"], positions["A"], shape_size, style_from, color)
        self._draw_arrow(draw, positions["arrow1"])
        self._draw_shape_at_position(draw, task_data["shape_b"], positions["B"], shape_size, style_to, color)
        
        self._draw_shape_at_position(draw, task_data["shape_c"], positions["C"], shape_size, style_from, color)
        self._draw_arrow(draw, positions["arrow2"])
        self._draw_question_mark(draw, positions["question"])
        
        return img
    
    def _render_final_state(self, task_data: Dict[str, Any]) -> Image.Image:
        """Render the final state with the answer revealed."""
        img = self.renderer.create_blank_image()
        draw = ImageDraw.Draw(img)
        
        width, height = self.config.image_size
        margin = self.config.margin
        shape_size = self.config.shape_size
        
        # Same layout as initial state
        positions = {
            "A": (margin + shape_size//2, height//4),
            "arrow1": (width//2, height//4),
            "B": (width - margin - shape_size//2, height//4),
            "C": (margin + shape_size//2, 3*height//4),
            "arrow2": (width//2, 3*height//4),
            "D": (width - margin - shape_size//2, 3*height//4)
        }
        
        # Draw shapes and arrows using task data styles and colors
        color = task_data["color"]
        style_from = task_data["style_from"]
        style_to = task_data["style_to"]
        
        self._draw_shape_at_position(draw, task_data["shape_a"], positions["A"], shape_size, style_from, color)
        self._draw_arrow(draw, positions["arrow1"])
        self._draw_shape_at_position(draw, task_data["shape_b"], positions["B"], shape_size, style_to, color)
        
        self._draw_shape_at_position(draw, task_data["shape_c"], positions["C"], shape_size, style_from, color)
        self._draw_arrow(draw, positions["arrow2"])
        self._draw_shape_at_position(draw, task_data["shape_d"], positions["D"], shape_size, style_to, color)  # Answer
        
        return img
    
    def _draw_shape_at_position(self, draw: ImageDraw.Draw, shape: str, position: Tuple[int, int], size: int, style: str, color: tuple = None):
        """Draw a shape at the specified position with the given style."""
        x, y = position
        
        # Use provided color or default
        if color is None:
            color = self.base_colors[0]  # Default to first color
        
        style_config = self.fill_styles[style]
        fill_color = color if style_config["fill"] else None
        outline_color = color
        outline_width = style_config["outline_width"]
        dash_pattern = style_config.get("dash", None)
        
        self._draw_base_shape(draw, shape, x, y, size, fill_color, outline_color, outline_width, dash_pattern)
    
    def _draw_base_shape(self, draw: ImageDraw.Draw, shape: str, x: int, y: int, size: int, fill_color, outline_color, outline_width: int, dash_pattern=None):
        """Draw a basic geometric shape with specified fill and outline."""
        half_size = size // 2
        
        # Note: PIL doesn't support dashed outlines directly for shapes, so we'll simulate it for simple shapes
        if shape == "square":
            draw.rectangle([x-half_size, y-half_size, x+half_size, y+half_size],
                         fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "circle":
            draw.ellipse([x-half_size, y-half_size, x+half_size, y+half_size], 
                        fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "triangle":
            points = [
                (x, y-half_size),  # top
                (x-half_size, y+half_size),  # bottom left
                (x+half_size, y+half_size)   # bottom right
            ]
            draw.polygon(points, fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "diamond":
            points = [
                (x, y-half_size),  # top
                (x+half_size, y),  # right
                (x, y+half_size),  # bottom
                (x-half_size, y)   # left
            ]
            draw.polygon(points, fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "pentagon":
            points = []
            for i in range(5):
                angle = i * 2 * math.pi / 5 - math.pi/2  # Start from top
                px = x + half_size * math.cos(angle)
                py = y + half_size * math.sin(angle)
                points.append((px, py))
            draw.polygon(points, fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "hexagon":
            points = []
            for i in range(6):
                angle = i * 2 * math.pi / 6
                px = x + half_size * math.cos(angle)
                py = y + half_size * math.sin(angle)
                points.append((px, py))
            draw.polygon(points, fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "rectangle":
            # Rectangle (wider than tall)
            width_factor = 1.4
            rect_width = int(half_size * width_factor)
            rect_height = int(half_size * 0.7)
            draw.rectangle([x-rect_width, y-rect_height, x+rect_width, y+rect_height], 
                         fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "oval":
            # Oval (wider than tall)
            width_factor = 1.4
            oval_width = int(half_size * width_factor)
            oval_height = int(half_size * 0.7)
            draw.ellipse([x-oval_width, y-oval_height, x+oval_width, y+oval_height], 
                        fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "star":
            # 5-pointed star
            points = []
            outer_radius = half_size
            inner_radius = half_size * 0.4
            
            for i in range(10):  # 5 outer + 5 inner points
                if i % 2 == 0:  # Outer points
                    angle = i * math.pi / 5 - math.pi/2
                    px = x + outer_radius * math.cos(angle)
                    py = y + outer_radius * math.sin(angle)
                else:  # Inner points
                    angle = i * math.pi / 5 - math.pi/2
                    px = x + inner_radius * math.cos(angle)
                    py = y + inner_radius * math.sin(angle)
                points.append((px, py))
            
            draw.polygon(points, fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "heart":
            # Heart shape using curves (approximate with polygon)
            points = [
                (x, y + half_size),                    # bottom point
                (x - half_size*0.7, y),              # left curve
                (x - half_size*0.3, y - half_size*0.5), # left top
                (x, y - half_size*0.2),              # center top
                (x + half_size*0.3, y - half_size*0.5),  # right top
                (x + half_size*0.7, y),               # right curve
            ]
            draw.polygon(points, fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "cross":
            # Cross shape
            thickness = half_size // 4
            # Vertical bar
            draw.rectangle([x-thickness, y-half_size, x+thickness, y+half_size],
                         fill=fill_color, outline=outline_color, width=outline_width)
            # Horizontal bar
            draw.rectangle([x-half_size, y-thickness, x+half_size, y+thickness],
                         fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "arrow":
            # Arrow pointing right
            points = [
                (x-half_size, y-half_size//2),  # left top
                (x, y-half_size//2),            # middle top
                (x, y-half_size),               # tip top
                (x+half_size, y),               # tip point
                (x, y+half_size),               # tip bottom
                (x, y+half_size//2),            # middle bottom
                (x-half_size, y+half_size//2)   # left bottom
            ]
            draw.polygon(points, fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "trapezoid":
            # Trapezoid (wider at bottom)
            top_width = half_size // 2
            points = [
                (x-top_width, y-half_size),     # top left
                (x+top_width, y-half_size),     # top right
                (x+half_size, y+half_size),     # bottom right
                (x-half_size, y+half_size)      # bottom left
            ]
            draw.polygon(points, fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "rhombus":
            # Rhombus (diamond with different proportions)
            points = [
                (x, y-half_size),               # top
                (x+half_size*0.7, y),           # right
                (x, y+half_size),               # bottom
                (x-half_size*0.7, y)            # left
            ]
            draw.polygon(points, fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "octagon":
            # Regular octagon
            points = []
            for i in range(8):
                angle = i * 2 * math.pi / 8
                px = x + half_size * math.cos(angle)
                py = y + half_size * math.sin(angle)
                points.append((px, py))
            draw.polygon(points, fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "crescent":
            # Crescent moon shape (two overlapping circles)
            # Draw larger circle
            draw.ellipse([x-half_size, y-half_size, x+half_size, y+half_size],
                        fill=fill_color, outline=outline_color, width=outline_width)
            # Draw smaller circle to create crescent (using background color)
            offset = half_size // 3
            smaller_radius = int(half_size * 0.7)
            draw.ellipse([x-smaller_radius+offset, y-smaller_radius, x+smaller_radius+offset, y+smaller_radius],
                        fill=(255,255,255), outline=outline_color, width=outline_width)
        
        elif shape == "plus":
            # Plus sign (thicker cross)
            thickness = half_size // 3
            # Vertical bar
            draw.rectangle([x-thickness, y-half_size, x+thickness, y+half_size],
                         fill=fill_color, outline=outline_color, width=outline_width)
            # Horizontal bar
            draw.rectangle([x-half_size, y-thickness, x+half_size, y+thickness],
                         fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "minus":
            # Minus sign (horizontal bar)
            thickness = half_size // 4
            draw.rectangle([x-half_size, y-thickness, x+half_size, y+thickness],
                         fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "L_shape":
            # L shape
            thickness = half_size // 3
            # Vertical part
            draw.rectangle([x-half_size, y-half_size, x-half_size+thickness, y+half_size],
                         fill=fill_color, outline=outline_color, width=outline_width)
            # Horizontal part
            draw.rectangle([x-half_size, y+half_size-thickness, x+half_size, y+half_size],
                         fill=fill_color, outline=outline_color, width=outline_width)
        
        elif shape == "T_shape":
            # T shape
            thickness = half_size // 3
            # Horizontal top part
            draw.rectangle([x-half_size, y-half_size, x+half_size, y-half_size+thickness],
                         fill=fill_color, outline=outline_color, width=outline_width)
            # Vertical part
            draw.rectangle([x-thickness//2, y-half_size, x+thickness//2, y+half_size],
                         fill=fill_color, outline=outline_color, width=outline_width)
    
    def _draw_arrow(self, draw: ImageDraw.Draw, position: Tuple[int, int]):
        """Draw a right-pointing arrow."""
        x, y = position
        length = self.config.arrow_length
        
        # Arrow shaft
        draw.line([x-length//2, y, x+length//2-10, y], fill=(0,0,0), width=3)
        
        # Arrow head
        points = [
            (x+length//2, y),
            (x+length//2-15, y-8),
            (x+length//2-15, y+8)
        ]
        draw.polygon(points, fill=(0,0,0))
    
    def _draw_question_mark(self, draw: ImageDraw.Draw, position: Tuple[int, int]):
        """Draw a question mark."""
        x, y = position
        size = self.config.question_mark_size
        
        try:
            font = ImageFont.truetype("arial.ttf", size)
        except:
            font = ImageFont.load_default()
        
        # Get text bounds for centering
        bbox = draw.textbbox((0, 0), "?", font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        text_x = x - w // 2
        text_y = y - h // 2
        
        draw.text((text_x, text_y), "?", font=font, fill=(100, 100, 100))
    
    # ══════════════════════════════════════════════════════════════════════════
    #  VIDEO GENERATION
    # ══════════════════════════════════════════════════════════════════════════
    
    def _generate_video(self, first_image: Image.Image, final_image: Image.Image, task_id: str, task_data: Dict[str, Any]) -> str:
        """Generate ground truth video showing the transformation."""
        temp_dir = Path(tempfile.gettempdir()) / f"{self.config.domain}_videos"
        temp_dir.mkdir(parents=True, exist_ok=True)
        video_path = temp_dir / f"{task_id}_ground_truth.mp4"
        
        # Create animation frames
        frames = self._create_transformation_frames(first_image, final_image, task_data)
        
        result = self.video_generator.create_video_from_frames(frames, video_path)
        return str(result) if result else None
    
    def _create_transformation_frames(self, first_image: Image.Image, final_image: Image.Image, task_data: Dict[str, Any], hold_frames: int = 15, morph_frames: int = 30) -> List[Image.Image]:
        """Create animation frames showing the fill-to-outline transformation."""
        frames = []
        
        # Hold initial state
        for _ in range(hold_frames):
            frames.append(first_image.copy())
        
        # Create fill-to-outline animation
        frames.extend(self._create_fill_to_outline_morph_frames(task_data, morph_frames))
        
        # Hold final state
        for _ in range(hold_frames):
            frames.append(final_image.copy())
        
        return frames
    
    def _create_fill_to_outline_morph_frames(self, task_data: Dict[str, Any], num_frames: int) -> List[Image.Image]:
        """Create frames showing the shape gradually changing from filled to outline."""
        frames = []
        
        width, height = self.config.image_size
        margin = self.config.margin
        shape_size = self.config.shape_size
        
        # Position of the shape that's being transformed (bottom right - the answer position)
        answer_x = width - margin - shape_size//2
        answer_y = 3*height//4
        
        shape_c = task_data["shape_c"]
        
        for i in range(num_frames):
            # Create frame with static elements
            img = self.renderer.create_blank_image()
            draw = ImageDraw.Draw(img)
            
            # Draw static elements (A, arrow, B, C, arrow)
            positions = {
                "A": (margin + shape_size//2, height//4),
                "arrow1": (width//2, height//4),
                "B": (width - margin - shape_size//2, height//4),
                "C": (margin + shape_size//2, 3*height//4),
                "arrow2": (width//2, 3*height//4),
            }
            
            # Get task color and styles
            color = task_data["color"]
            style_from = task_data["style_from"]
            style_to = task_data["style_to"]
            
            # Draw ALL static shapes - these NEVER change during animation
            self._draw_shape_at_position(draw, task_data["shape_a"], positions["A"], shape_size, style_from, color)
            self._draw_arrow(draw, positions["arrow1"])
            self._draw_shape_at_position(draw, task_data["shape_b"], positions["B"], shape_size, style_to, color)
            self._draw_shape_at_position(draw, task_data["shape_c"], positions["C"], shape_size, style_from, color)
            self._draw_arrow(draw, positions["arrow2"])
            
            # ONLY the answer shape changes during animation
            # Interpolate between style_from and style_to
            progress = i / (num_frames - 1) if num_frames > 1 else 1.0
            
            # Create intermediate style based on transformation type
            if progress < 0.5:
                # First half: use from style
                current_style = style_from
            else:
                # Second half: use to style
                current_style = style_to
            
            self._draw_shape_at_position(draw, shape_c, (answer_x, answer_y), shape_size, current_style, color)
            
            frames.append(img)
        
        return frames
