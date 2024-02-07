from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image
import configparser
import numpy as np
from io import BytesIO
import keras

config = configparser.ConfigParser()
config.read('config.ini')
saved_model = config['OUTPUT_PATH']['MODEL']
st.title('Skin Malignant Detection')

if 'model' not in st.session_state:
	st.session_state.model = None
if 'input_image' not in st.session_state:
	st.session_state.input_image = None


@st.cache_data(persist='disk')
def model_init():
	try:
		st.session_state.model = keras.models.load_model(saved_model)
	except Exception as e:
		st.write('Model files are missing')


def process_input(input_image):
	stream = BytesIO(input_image.getvalue())
	image = Image.open(stream)
	st.session_state.input_image = image.resize((256, 256))
	st.subheader('Input image')
	st.image(st.session_state.input_image)
	

def predict_input():
	y_hat = st.session_state.model.predict(np.expand_dims(st.session_state.input_image, 0))
	if y_hat < 0.5:
		st.success(f"Output : Benign")
	else:
		st.error(f"Output : Malignant")


def main():
	model_init()
	with st.sidebar:	
		input_file = st.file_uploader('Upload the image', type = ['jpeg','jpg','png'])
		col1, col2 = st.columns(2)
		with col1:
			sumitBtn = st.button('Process Image')
		with col2:
			cacheBtn = st.button('Clear cache')
			if cacheBtn:
				st.cache_data.clear()

	if sumitBtn:
		input_image = process_input(input_file)
		predict_input()



if __name__ == '__main__':
	main()
