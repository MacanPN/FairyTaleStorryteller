import streamlit as st
from PIL import Image
import base64
from io import BytesIO
import requests
import openai
import json

app_description = """
Welcome to the Enchanted Story Weaver, where dreams and drawings intertwine to create magical fairy tales!

**How It Works:**
1. **Upload Your Drawing:** Share your imaginative artwork with our mystical fairies.
2. **Provide Clues (Optional):** Offer hints or let the fairies work their magic solo.
3. **Experience the Magic:** Sit back and watch as our fairies transform your drawing into an enchanting story.
🧚✨
"""

st.sidebar.markdown(app_description)
api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password", help="You can find your OpenAI API on the [OpenAI dashboard](https://platform.openai.com/account/api-keys)")

model_text_completion = "gpt-4" # gpt-3.5-turbo-0125
model_image_2_text = "gpt-4-vision-preview"
model_text_2_image = "dall-e-3"

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
    print(api_key)
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

    print(response.json())
    if "error" in response.json():
        st.error(response.json())
        return False
    img_desc = response.json()["choices"][0]["message"]["content"]
    return img_desc

def do_magic(input_image, story_clue, client):
    # analyze the input image
    with st.spinner("Your drawing has caught the eye of our band of mystical fairies! They're diving into the intricate details of your creation, ready to craft a fairy tale filled with magic and wonder. Stay tuned for the enchanting reveal!... ⏳"):
        prompt = "What’s in this image? Only describe the people, objects and scenery and don't comment on the style or technique used."
        img_desc = image2text(base64_image=input_image, prompt=prompt)
        characters_prompt = """
        Below is pasted a description of an image. Extract from it only description of characters: people and animals (if any). 
        Ignore the scenary, surroundings, pose etc. Include what clothes and shoes they are wearing, including their color; their ethnicity, hair color etc.

        Image description: {img_desc}
        """
        characters_response = client.chat.completions.create(
        model=model_text_completion,
        messages=[
            {"role": "system", "content": "You are a helpful asistant."},
            {"role": "user", "content": characters_prompt.format(img_desc=img_desc)}
        ],
        )
        characters = characters_response.choices[0].message.content
        print("Characters:",characters,"\n\n")
    st.write("Hidden wonders revealed by fairy light ✅")
    # generate story
    with st.spinner("Within the labyrinth of dreams, stories weave their intricate threads. Be patient, for soon the tapestry of magic shall be revealed 🌙✨"):
        print(img_desc)
        fairy_tale_user_prompt = img_desc
        if len(story_clue.strip())>5:
            fairy_tale_user_prompt += "Please also incorporate following clue(s) to the story line: {story_clue}".format(story_clue=story_clue)
        fairy_tale_sys_prompt = """ You are an automated system that generates a short fairy tale from an image description. 
            User sends description of an image and you make use of the described characters, animals, objects and scenary in general, and write a fairy tale in 4 paragraphs.
            Ignore information about the style and capabilities of the image author.
            Describe a lively story with a touch of magic. Make up other characters, people and/or animals and have them interact with the main character.
            Each paragraph should be ~100 words.

            Return result in JSON array format containing 4 elements. Each paragraph will be an element in the array.
        """
        story = client.chat.completions.create(
            model=model_text_completion,
            messages=[
                {"role": "system", "content": fairy_tale_sys_prompt},
                {"role": "user", "content": fairy_tale_user_prompt}
            ]
        )
        paragraphs = eval(story.choices[0].message.content)
        print("Story:")
        for i,p in enumerate(paragraphs):
            print("\t",i,":",p)
    st.write("Pages flutter as tales come alive ✅")

    # come up with the name for the fairy tale
    with st.spinner("Just a moment longer! We're weaving spells to find the ideal title for your fairy tale. Get ready to be captivated by the magic of words!"):
        naming_prompt = """Come up with a fitting name for the following fairy tale:
        
        Story: {story}
        """
        story_title = client.chat.completions.create(
            model=model_text_completion,
            messages=[
                {"role": "user", "content": naming_prompt.format(story="\n".join(paragraphs))}
            ]
        ).choices[0].message.content.strip('"')
        print("Title:",story_title)
    st.write("Words shimmer, forming the perfect enchantment ✅")    

    # generate illustration descriptions
    with st.spinner("The magic doesn't stop with words! We're in the midst of conjuring beautiful illustrations to bring your fairy tale to life. Your patience will soon be rewarded!"):
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
        
        Individuals: {characters}

        Try to immitate animation style of Disney or Pixar studios.
        """
        output_images = []
        for i,image_desc in enumerate(image_descriptions):
            output_images.append(
                Image.open(
                    BytesIO(
                        base64.b64decode(
                            client.images.generate( model=model_text_2_image,
                                                    prompt=image_prompt.format(img_desc=image_desc, characters=characters),
                                                    size="1024x1024",
                                                    quality="standard",
                                                    response_format="b64_json",
                                                    n=1,
                                                    ).data[0].b64_json
                        )
                    )
                )
            )
    st.write("Canvas aglow with mystical strokes ✅")
    return {"input_img_desc":img_desc,
            "story_title": story_title,
            "story": paragraphs,
            "output_imgs_desc": image_descriptions,
            "output_images": output_images
    }

def main():
    st.title("Enchanted Story Weaver")
    client = openai.OpenAI(api_key=api_key)
    uploaded_file = st.sidebar.file_uploader("Summon the Story: Share Your Mystical Art", type=["jpg", "jpeg", "png"])
    story_clue = st.sidebar.text_area("Craft the Journey: Provide Story Sparks!\nDon't worry about filling this space! Our mystical fairies can also weave a tale solely from the essence of your drawing.",
                                      height=40,
                                      max_chars=500)

    if uploaded_file is not None:
        st.subheader("Thank you for this masterpiece!", divider="blue")
        image = Image.open(uploaded_file)
        image = scale_image(image)
        print(type(image))
        st.image(image, caption="Uploaded Image", width=300)

        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
    if st.sidebar.button("Whisper magic!"):
        magic = do_magic(image_base64, story_clue, client)
        
        st.header(magic["story_title"])
        for i,p in enumerate(magic["story"]):
            st.write(p)
            st.image(magic["output_images"][i])
        
    

if __name__ == "__main__":
    main()
