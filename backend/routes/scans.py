from flask import Blueprint, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from backend.extensions import db
from backend.models.scan import Scan

scans_bp = Blueprint('scans', __name__)


@scans_bp.route('/scans/', methods=['GET'])
@jwt_required()
def get_scans():
    user_id = int(get_jwt_identity())
    scans = (
        Scan.query
        .filter_by(user_id=user_id)
        .order_by(Scan.uploaded_at.desc())
        .all()
    )
    return jsonify([s.to_dict() for s in scans]), 200


@scans_bp.route('/scans/<int:scan_id>', methods=['GET'])
@jwt_required()
def get_scan(scan_id):
    user_id = int(get_jwt_identity())
    scan = Scan.query.filter_by(id=scan_id, user_id=user_id).first()
    if not scan:
        return jsonify({'error': 'Scan not found'}), 404
    return jsonify(scan.to_dict()), 200
