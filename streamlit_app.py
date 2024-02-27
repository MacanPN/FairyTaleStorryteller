import streamlit as st
from PIL import Image
import base64
import lorem
from io import BytesIO
import requests
import openai
import json

model_text_completion = "gpt-4" # gpt-3.5-turbo-0125
model_image_2_text = "gpt-4-vision-preview"
model_text_2_image = "dall-e-3"

api_key = "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"
#api_key = ""
client = openai.OpenAI()

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Function to scale down the image if it's larger than 1080p
def scale_image(image):
    width, height = image.size
    if width > 1920 or height > 1080:
        aspect_ratio = width / height
        if aspect_ratio > 16/9:  # If width is the limiting factor
            new_width = 1920
            new_height = int(new_width / aspect_ratio)
        else:  # If height is the limiting factor
            new_height = 1080
            new_width = int(new_height * aspect_ratio)
        image = image.resize((new_width, new_height))
    return image

def image2text(image_path=None, base64_image=None, prompt="What’s in this image?", max_tokens=300):
    if image_path:
        base64_image = encode_image(image_path)
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": model_image_2_text,
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": prompt
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    #print(response.json())
    img_desc = response.json()["choices"][0]["message"]["content"]
    return img_desc

def do_magic(input_image):
    # analyze the input image
    prompt = "What’s in this image? Only describe the people, objects and scenery and don't comment on the style or technique used."
    img_desc = image2text(base64_image=input_image, prompt=prompt)

    # generate story
    print(img_desc)
    fairy_tale_prompt = """ You are an automated system that generates a short fairy tale from an image description. 
        User sends description of an image and you make use of the described characters, animals, objects and scenary in general, and write a fairy tale in 4 paragraphs.
        Ignore information about the style and capabilities of the image author.
        Describe a lively story with a touch of magic. Make up other characters, people and/or animals and have them interact with the main character.
        Each paragraph should be ~100 words.

        Return result in JSON array format containing 4 elements. Each paragraph will be an element in the array.
    """
    story = client.chat.completions.create(
        model=model_text_completion,
        messages=[
            {"role": "system", "content": fairy_tale_prompt},
            {"role": "user", "content": img_desc}
        ]
    )
    paragraphs = eval(story.choices[0].message.content)
    print("Story:")
    for i,p in enumerate(paragraphs):
        print("\t",i,":",p)
    
    # come up with the name for the fairy tale
    naming_prompt = """Come up with a fitting name for the following fairy tale:
    
    Story: {story}
    """
    story_title = client.chat.completions.create(
        model=model_text_completion,
        messages=[
            {"role": "user", "content": naming_prompt.format(story="\n".join(paragraphs))}
        ]
    ).choices[0].message.content
    print("Title:",story_title)
    # generate illustration descriptions
    image_description_sys_prompt = """
    You are an assistant that generates ideas for fairy tale illustrations.
    User submits a fairy tale and you generate description of one illustration per paragraph.

    Return the results in a JSON array. Each array element will be string describing one illustration.
    """
    image_description_user_prompt = """
    Following fairy tale consists of 4 paragraphs. Generate illustration ideas for each paragraph.

    Fairy tale: {fairy_tale}
    """
    image_description_response = client.chat.completions.create(
        model=model_text_completion,
        messages=[
            {"role": "system", "content": image_description_sys_prompt},
            {"role": "user", "content": image_description_user_prompt.format(fairy_tale="\n\n".join(paragraphs))}
        ]
    )
    image_descriptions = json.loads(image_description_response.choices[0].message.content)
    print("Image descriptions:")
    for i,p in enumerate(image_descriptions):
        print(i,":",p)
    
    # generate illustrations
    image_prompt = """
    Image description: {img_desc}

    Try to immitate animation style of Disney or Pixar studios.
    """
    output_images = []
    for i,image_desc in enumerate(image_descriptions):
        output_images.append(
            Image.open(
                BytesIO(
                    base64.b64decode(
                        client.images.generate( model=model_text_2_image,
                                                prompt=image_prompt.format(img_desc=image_desc),
                                                size="1024x1024",
                                                quality="standard",
                                                response_format="b64_json",
                                                n=1,
                                                ).data[0].b64_json
                    )
                )
            )
        )
    
    return {"input_img_desc":img_desc,
            "story_title": story_title,
            "story": paragraphs,
            "output_imgs_desc": image_descriptions,
            "output_images": output_images
    }

def main():
    st.title("MagicCanvas: Where Drawings Spark Enchanting Stories")
    api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password", help="You can find your OpenAI API on the [OpenAI dashboard](https://platform.openai.com/account/api-keys)")

    uploaded_file = st.sidebar.file_uploader("Upload your drawing", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.header("Your image")
        image = Image.open(uploaded_file)
        image = scale_image(image)
        print(type(image))
        st.image(image, caption="Uploaded Image", width=300)

        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        magic = do_magic(image_base64)
        st.header(magic["story_title"])
        for i,p in enumerate(magic["story"]):
            st.write(p)
            st.image(magic["output_images"][i])
        
    

if __name__ == "__main__":
    main()
