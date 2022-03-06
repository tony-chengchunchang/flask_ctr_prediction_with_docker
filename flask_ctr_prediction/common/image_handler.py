ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def is_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
