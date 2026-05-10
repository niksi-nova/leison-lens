from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from backend.extensions import db
from backend.models.scan import Scan
from backend.ml.inference import predict

predict_bp = Blueprint('predict', __name__)


@predict_bp.route('/predict', methods=['POST'])
@jwt_required()
def run_predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    patient_name = (request.form.get('patient_name') or '').strip()
    patient_age_raw = request.form.get('patient_age')
    patient_age = None
    if patient_age_raw:
        try:
            patient_age = int(patient_age_raw)
        except ValueError:
            pass

    image_bytes = image_file.read()
    try:
        result = predict(image_bytes)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    user_id = int(get_jwt_identity())
    scan = Scan(
        user_id=user_id,
        patient_name=patient_name or 'Unknown Patient',
        patient_age=patient_age,
        filename=image_file.filename or 'upload',
        grade=result['grade'],
        grade_label=result['grade_label'],
        confidence=result['confidence'],
        prob_ma=result['lesions']['MA'],
        prob_he=result['lesions']['HE'],
        prob_ex=result['lesions']['EX'],
        prob_se=result['lesions']['SE'],
    )
    db.session.add(scan)
    db.session.commit()

    return jsonify({
        **result,
        'scan_id': scan.id,
    }), 200
