from tenacity import retry, stop_after_attempt, wait_exponential
import re
from typing import List, Dict, Optional, Any
import anthropic
import fal_client

CLAUDE_KEY = "XXXX"

FAL_KEY = "XXXX"

IMAGE_GEN_SYSTEM_PROMPT = """You are an AI assistant tasked with creating an ideal image generation prompt for a given lesson. Your goal is to craft a detailed and effective prompt that will result in a high-quality, visually striking cover image for a lesson.

Consider these general guidelines that should be applied to all image prompts:
<general_guidelines>
1. **Be Specific and Detailed**. Include precise details such as setting, objects, colors, mood, and specific elements you want in the image.
2. **Use Descriptive Language**. Employ clear, visual descriptions and avoid abstract terms to help the AI better understand and generate the desired image.
3. **Define the Mood or Atmosphere**. Use mood descriptors like "serene," "chaotic," "mystical", etc. to set the emotional tone of the image.
4. **Specify Composition and Perspective**. Mention desired perspectives such as close-up, wide shot, or specific angles to frame the scene correctly.
5. **Detail Lighting and Time of Day**. Clarify lighting conditions (e.g., sunny, candlelight) and time of day to influence the image's mood and visibility.
6. **Describe Actions or Movements**. If dynamic imagery is desired, describe actions or movements to add energy to the scene.
7. **Avoid Overloading the Prompt**. Provide enough detail to guide the AI but avoid excessive specifics that could confuse the model.
8. **Use Commas to Separate Elements**. Clearly separate different elements of your prompt with commas to help the AI distinguish between them.
9. **Focus on Positive Descriptions**. Concentrate on what you want to include rather than what you want to exclude, unless using a platform that supports negative prompts effectively.
</general_guidelines>

Additionally, here are optional guidelines that should be used when appropriate:
<optional_guidelines>
1. **Incorporate Artistic Styles**. Specify desired artistic styles or themes, such as "impressionist" or "photorealistic",  to guide the AI's style interpretation.
2. **Use Analogies or Comparisons**. Employ comparisons or analogies to familiar scenes or styles for clearer interpretation (e.g., "like Van Gogh's Starry Night").
3. **Employ Photographic Terminologies**. Use terms related to photography like "low angle," "backlighting," or "shallow depth of field" to specify visual details.
4. **Utilize Artistic References**. Reference specific artists or art movements to infuse their techniques or aesthetics into your image.
5. **Adjust Prompt Weighting**. Use syntax or symbols specific to the AI tool to emphasize or de-emphasize elements (e.g., "mountains::2" to emphasize).
6. **Experiment with Prompt Length**. Test different prompt lengths to see how they affect the detail and style of the generated images.
7. **Specify Environmental Context**. Include details about the environment or background to place the subject in a more defined space.
8. **Use Platform-Specific Features**. Leverage unique features of the AI platform, like aspect ratios or style modifiers, to enhance results.
9. **Combine Text and Image Prompts**. When possible, use a combination of text and reference images to provide clear direction and inspiration to the AI.
</optional_guidelines>

Examples:
- See how these prompts focus on only the critical subject and avoid overloading the prompt with excessive details. You to should, avoid extravagance or overly vivid descriptions that are insignificant to the scene's message and purpose but add unnecessary complexity to the prompt. 
- IMPORTANT: The prompt should be extremely succinct, 1-3 sentences maximum. This must be always followed, the prompt can't be too detailed and long otherwise it will produce terir
<examples>
1. **Sunset Beach Scene**:
   - "Create a photorealistic image of a serene beach at sunset, with soft golden light reflecting on calm waters. Include a silhouette of a lone person walking along the shore, the sky painted with hues of orange and pink. Use a wide shot to capture the expansive horizon."

2. **Futuristic Cityscape**:
   - "Generate a high-resolution image of a bustling cyberpunk cityscape at night. The scene should feature towering skyscrapers with neon lights, flying cars weaving between buildings, and pedestrians in futuristic attire. Emphasize the vibrant blues and pinks of the neon signs and include a rainy atmosphere to enhance the mood."

3. **Portrait in Van Gogh's Style**:
   - "Create an impressionist portrait of a young woman, styled in the manner of Van Gogh. She should have expressive blue eyes and curly red hair, wearing a green vintage dress. The background should be a swirl of vibrant colors that evoke a sense of movement, with a focus on brushstroke textures."

4. **Wildlife Photography - Majestic Lion**:
   - "Produce a photorealistic image of a majestic lion standing on a rocky outcrop during the golden hour. The lighting should highlight the intricate details of the lion's mane and the intense gaze in its eyes. Include a savannah landscape in the background with acacia trees and a soft-focus effect to emphasize the subject."

5. **Medieval Knight Scene**:
   - "Illustrate a dramatic scene featuring a medieval knight in shiny armor, holding a sword, standing in a misty forest. The image should capture the early morning light filtering through the trees, creating a play of light and shadow. Use a low angle to give a heroic feel to the knight, and include detailed textures on the armor and sword."
</examples>
"""

IMAGE_GEN_USER_PROMPT = """### Further Instructions
I will provide you with a the name of the lesson and you should write the image generation prompt for the cover image of the lesson. The image should realistically try to encapsulate the essence of the lesson.
 - Incorporate these guidelines into your prompt as appropriate. Be sure to use descriptive language that captures the mood, lighting, composition, and key elements of the scene. Include relevant artistic styles, camera angles, or other technical details that would enhance the image.
 - When deciding between different options for mood, composition, perspective, and time of day in AI-generated images, consider the narrative purpose, emotional impact, and visual aesthetics you aim to achieve. Evaluate how each element aligns with the overall theme and message of the image.
 - Craft your image prompt in a clear, succinct manner that will guide an image generation AI to create the most compelling Photo-Realistic visualization of the scene. Aim for a prompt that is detailed enough to capture the essence of the scene but not so complex that it becomes confusing.
 - Please provide your final image generation prompt within <image_prompt> tags. Your prompt should be a single, well-structured paragraph without any line breaks.
 - These images are to be presented as part of an SAT Prep course. 

<lesson_name>
{description}
</lesson_name>
"""


def claude_complete(messages):
    client = anthropic.Anthropic(
        api_key=CLAUDE_KEY,
    )

    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4000,
        temperature=1,
        system=messages[0],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": messages[1]
                    }
                ]
            }
        ]
    )
    return message.content[0].text

def generate_img_prompt(description: str) -> str:
    messages = [
        IMAGE_GEN_SYSTEM_PROMPT,
        IMAGE_GEN_USER_PROMPT.format(description=description)
    ]

    prompt = claude_complete(messages)
    
    extracted_prompt = extract_tag_content('image_prompt', prompt)
    if extracted_prompt:
        return extracted_prompt
    else:
        return prompt

def extract_tag_content(tag_name: str, text: str)->Optional[str]:
    pattern = f"<{tag_name}[^>]*>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_image(prompt: str) -> Dict[str, Any]:
    handler = fal_client.submit(
        "fal-ai/flux-pro",
        arguments={
            "prompt": prompt,
            "image_size": {
                "width": 1024,
                "height": 512
            }
        },
    )
    return handler.get()

def get_lesson_descriptions() -> List[str]:
    # TODO: Add lesson descriptions here
    return [
        "Reading and Writing: Information and Ideas"
        # "Reading and Writing: Craft and Structure",
        # "Reading and Writing: Expression of Ideas",
        # "Reading and Writing: Standard English Conventions",
        # "Math: Algebra",
        # "Math: Advanced Math",
        # "Math: Problem Solving and Data Analysis",
        # "Math: Geometry and Trigonometry"
    ]

def generate_images_for_lessons():
    lesson_descriptions = get_lesson_descriptions()
    generated_urls = []
    
    for description in lesson_descriptions:
        print(f"Generating image for: {description}")
        
        img_prompt = generate_img_prompt(description)
        print(f"Image prompt: {img_prompt}")
        
        try:
            result = generate_image(img_prompt)
            image_url = result.get('images', [{}])[0].get('url')
            if image_url:
                print(f"Image generated successfully. URL: {image_url}")
                generated_urls.append((description, image_url))
            else:
                print("Failed to generate image: No URL in the response")
        except Exception as e:
            print(f"Error generating image: {str(e)}")
        
        print("\n" + "-"*50 + "\n")
    
    print("All generated image URLs:")
    for description, url in generated_urls:
        print(f"{url}")

if __name__ == "__main__":
    generate_images_for_lessons()