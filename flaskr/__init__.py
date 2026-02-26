import os
import flask as fl
import io
import generator
import librosa
import numpy

def create_app(test_config=None):
    # create and configure the app
    app = fl.Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
        MAX_CONTENT_LENGTH=50 * 1024 * 1024,  # 50MB max file size (increased from 16MB)
        UPLOAD_FOLDER='uploads'
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    @app.route('/')
    def home():
        return fl.render_template('index.html')
    
    @app.route('/test')
    def test_page():
        return fl.render_template('test.html')
    
    @app.route('/test-upload', methods=['POST'])
    def test_upload():
        return fl.jsonify({
            'method': fl.request.method,
            'files': list(fl.request.files.keys()),
            'form_keys': list(fl.request.form.keys()),
            'content_type': fl.request.content_type,
            'has_file': 'file' in fl.request.files,
            'file_data': str(fl.request.files.get('file')) if 'file' in fl.request.files else None
        })

    @app.route('/upload', methods=['POST'])
    def upload():
        try:
            print(f"DEBUG: Request method: {fl.request.method}")
            print(f"DEBUG: Content type: {fl.request.content_type}")
            print(f"DEBUG: Content length: {fl.request.content_length}")
            print(f"DEBUG: Files in request: {list(fl.request.files.keys())}")
            print(f"DEBUG: Form keys: {list(fl.request.form.keys())}")
            print(f"DEBUG: Request values: {dict(fl.request.values)}")
            print(f"DEBUG: Request files dict: {dict(fl.request.files)}")
            
            # Check for file size limit error
            if fl.request.content_length and fl.request.content_length > app.config['MAX_CONTENT_LENGTH']:
                return fl.jsonify({'error': f'File too large. Maximum size is {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB'}), 413
            
            # Check if file exists in request
            if 'file' not in fl.request.files:
                return fl.jsonify({'error': 'No file field in request', 'debug': {
                    'files': list(fl.request.files.keys()), 
                    'form': list(fl.request.form.keys()), 
                    'content_type': fl.request.content_type,
                    'content_length': fl.request.content_length,
                    'all_values': dict(fl.request.values),
                    'files_dict': dict(fl.request.files)
                }}), 400
                
            file = fl.request.files['file']
            print(f"DEBUG: File object: {file}")
            print(f"DEBUG: File filename: {file.filename if file else 'None'}")
            
            if file and file.filename:
                # Get customization parameters from form
                size = int(fl.request.form.get('size', 1080))
                vibrancy = float(fl.request.form.get('vibrancy', 1.0))
                gradient_type = fl.request.form.get('gradient', 'linear')
                
                # Map resolution to appropriate width
                resolution_width_map = {
                    720: 1280,   # 720p HD (16:9 aspect ratio)
                    1080: 1920,  # 1080p Full HD (16:9 aspect ratio)
                    1440: 2560,  # 1440p 2K (16:9 aspect ratio)
                    2160: 3840   # 2160p 4K (16:9 aspect ratio)
                }
                width = resolution_width_map.get(size, 1920)
                
                # Validate parameters
                if size < 480 or size > 3840:
                    size = 1080
                if vibrancy < 0.0 or vibrancy > 2.0:
                    vibrancy = 1.0
                if gradient_type not in ['linear', 'radial']:
                    gradient_type = 'linear'
                
                # Create a file-like object from uploaded file
                file_data = io.BytesIO(file.read())
                file_data.seek(0)  # Reset pointer to beginning
                
                audio, sr = librosa.load(file_data, duration=30)
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
                norm = ((mfcc - numpy.min(mfcc)) / (numpy.max(mfcc) - numpy.min(mfcc))) * 255
                img = generator.create_gradient_wallpaper(norm, size, vibrancy, width, gradient_type)
                img_io = io.BytesIO()
                img.save(img_io, format='PNG')
                img_io.seek(0)
                return fl.send_file(img_io, mimetype='image/png')
            else:
                return fl.jsonify({'error': 'No file provided or file is empty', 'debug': {'filename': file.filename if file else None}}), 400
        except Exception as e:
            print(f"DEBUG: Exception: {str(e)}")
            return fl.jsonify({'error': f'Processing failed: {str(e)}'}), 500   

    return app
