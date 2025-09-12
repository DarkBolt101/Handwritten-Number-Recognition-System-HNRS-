import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

class GrayscaleNumberProcessor:
    def __init__(self):
        print("Grayscale")
        print("=" * 50)
        
    def create_sample_image(self, save_path="numbers.png"):
        """Create a sample image with numbers for demonstration"""
        # Create a blank image with white background
        img = Image.new('RGB', (300, 150), color='white')
        draw = ImageDraw.Draw(img)
        
        # fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()
        
        # Draw some numbers with different colors
        draw.text((20, 50), "123", fill=(255, 0, 0), font=font)  # Red
        draw.text((110, 50), "456", fill=(0, 128, 0), font=font)  # Green
        draw.text((200, 50), "789", fill=(0, 0, 255), font=font)  # Blue
        
        # Save the image
        img.save(save_path)
        print(f"Sample image created: {save_path}")
        return np.array(img)
    
    def convert_to_grayscale(self, image_array, method='luminance'):
        """Convert RGB image to grayscale using different methods"""
        if method == 'luminance':
            # Perceptual grayscale: Y = 0.299*R + 0.587*G + 0.114*B
            grayscale = np.dot(image_array[..., :3], [0.299, 0.587, 0.114])
        elif method == 'average':
            # Simple average of RGB channels
            grayscale = np.mean(image_array[..., :3], axis=2)
        elif method == 'lightness':
            # Average of max and min RGB values
            grayscale = (np.max(image_array[..., :3], axis=2) + 
                         np.min(image_array[..., :3], axis=2)) / 2
        else:
            raise ValueError("Method must be 'luminance', 'average', or 'lightness'")
        
        return grayscale.astype(np.uint8)
    
    def display_results(self, original, grayscale, method_name):
        """Display the original and grayscale images side by side"""
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(original)
        plt.title('Original Color Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(grayscale, cmap='gray')
        plt.title(f'Grayscale ({method_name} method)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_grayscale_methods(self):
        """Demonstrate different grayscale conversion methods"""
        # Create a sample image
        original_image = self.create_sample_image()
        
        # Apply different grayscale methods
        methods = [
            ('luminance', 'Luminance (0.299R + 0.587G + 0.114B)'),
            ('average', 'Average (R+G+B)/3'),
            ('lightness', 'Lightness (max+min)/2')
        ]
        
        for method, method_name in methods:
            print(f"\nApplying {method_name} method...")
            grayscale_image = self.convert_to_grayscale(original_image, method)
            self.display_results(original_image, grayscale_image, method_name)
            
            # Print some pixel values for analysis
            print("Sample pixel values (original RGB -> grayscale):")
            for i in range(3):
                rgb_val = original_image[75, 50 + i*90, :]
                gray_val = grayscale_image[75, 50 + i*90]
                print(f"  Pixel at (75, {50 + i*90}): RGB{rgb_val} -> {gray_val}")
    
    def explain_grayscale(self):
        """Explain the importance of grayscale for number recognition"""
        print("\n" + "="*60)
        print("WHY GRAYSCALE IS IMPORTANT FOR NUMBER RECOGNITION")
        print("="*60)
        print("1. Reduces Complexity: 3 color channels → 1 intensity channel")
        print("2. Improves Processing Speed: Less data to process")
        print("3. Enhances Contrast: Makes number features more prominent")
        print("4. Removes Color Variance: Focuses on shape rather than color")
        print("5. Standardizes Input: Consistent input for recognition algorithms")
        print("6. Reduces Memory Usage: 1/3 of the original data size")
        
        print("\nGrayscale conversion formulas:")
        print("  Luminance: 0.299*R + 0.587*G + 0.114*B (most accurate perceptually)")
        print("  Average:   (R + G + B) / 3 (simplest method)")
        print("  Lightness: (max(R,G,B) + min(R,G,B)) / 2 (preserves dynamics)")

# Run the demonstration
if __name__ == "__main__":
    processor = GrayscaleNumberProcessor()
    processor.demonstrate_grayscale_methods()
    processor.explain_grayscale()