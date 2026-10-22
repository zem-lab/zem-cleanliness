"""Resolve full-frame ambiguity only with an explicitly bounded calibration prior.

This is a location constraint, never evidence that a model contour is correct.
The caller must still validate the observed contour, template overlap and EDS point.
"""
from __future__ import annotations

import math
import cv2
import numpy as np


def _valid_axes(axes):
    if axes.shape!=(2,2) or not np.isfinite(axes).all():
        return False
    # Reuse the existing measured-axis calibration scale limits. Calibrated
    # axes may contain small anisotropy/coupling and are not necessarily orthogonal.
    singular=np.linalg.svd(axes,compute_uv=False)
    return bool(singular.min()>=.7 and singular.max()<=1.3)


def validate_selected_location(image_shape, particle, pixel_size_um, center, *,
                               calibration_location, stage_center_um,
                               field_rotation_deg=0., pixel_axis_matrix=((1.,0.),(0.,1.)), measured_contour=None):
    """A unique shape score cannot move acquisition outside the calibrated ROI."""
    result={"accepted":False,"allowsEds":False,"method":"calibrated_selected_center_guard_v1"}
    try:
        if calibration_location.get('reviewAccepted') is not True:
            return {**result,'reason':'标定尚未确认'}
        h,w=image_shape[:2]
        radius=float(calibration_location['maximumCenterErrorUm'])
        target=np.asarray(calibration_location['targetStageUm'],float)
        stage=np.asarray(stage_center_um,float);point=np.asarray(center,float)
        axes=np.asarray(pixel_axis_matrix,float)
        if (target.shape!=(2,) or stage.shape!=(2,) or point.shape!=(2,) or axes.shape!=(2,2)
                or not np.isfinite([target,stage,point]).all() or not np.isfinite(axes).all()
                or not math.isfinite(pixel_size_um) or pixel_size_um<=0
                or not math.isfinite(field_rotation_deg) or not math.isfinite(radius)
                or not 0<radius<=min(h,w)*pixel_size_um*.2
                or not _valid_axes(axes)
                or np.any(point<0) or np.any(point>=np.asarray([w,h]))):
            return {**result,'reason':'标定位置约束或候选坐标无效'}
        if np.linalg.norm(target-stage)>2.:
            return {**result,'reason':'采图台位置偏离预测位置，不能使用该图采谱'}
        angle=math.radians(field_rotation_deg)
        rotation=np.asarray([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        observed=stage+rotation@axes@(point-np.asarray([w/2,h/2]))*pixel_size_um
        distance=float(np.linalg.norm(observed-target))
        # Optical reference, template-box center and measured centroid need not
        # coincide. Require the observed particle to intersect the calibrated
        # region, not an invented equality of these distinct point definitions.
        if measured_contour is None:
            extent=float(particle.length_um)/2
            if not math.isfinite(extent) or extent<=0:
                return {**result,'reason':'光镜颗粒尺寸无效'}
            support_distance=max(0.,distance-extent)
            support_source='optical_half_length_precheck'
        else:
            contour=np.asarray(measured_contour,np.float32)
            if (contour.ndim!=2 or contour.shape[1:]!=(2,) or len(contour)<3
                    or not np.isfinite(contour).all() or np.any(contour<0)
                    or np.any(contour[:,0]>=w) or np.any(contour[:,1]>=h)):
                return {**result,'reason':'实测轮廓坐标无效'}
            physical_contour=(contour-np.asarray([w/2,h/2]))@(rotation@axes*pixel_size_um).T+(stage-target)
            signed=cv2.pointPolygonTest(physical_contour.astype(np.float32),(0.,0.),True)
            support_distance=max(0.,-signed)
            support_source='measured_sem_contour'
        result.update(maximumCenterErrorUm=radius,candidateDistanceUm=distance,
                      distanceToParticleUm=support_distance,supportSource=support_source,
                      source='task_calibration_acceptance_maxErrorUm')
        return {**result,'accepted':support_distance<=radius,
                'reason':'候选颗粒与标定允许范围相交' if support_distance<=radius else '候选颗粒完全位于标定允许范围之外，不能采谱'}
    except (KeyError,ValueError,TypeError):
        return {**result,'reason':'标定位置约束信息不完整或无效'}


def review_calibrated_location(image_shape, particle, pixel_size_um, audit, *,
                               calibration_location, stage_center_um,
                               field_rotation_deg=0., pixel_axis_matrix=((1.,0.),(0.,1.)),
                               minimum_template_score=None):
    result = {"accepted":False, "allowsEds":False, "method":"bounded_calibration_location_v1"}
    def reject(reason):
        return {**result,"reason":reason}
    if not calibration_location or calibration_location.get("reviewAccepted") is not True:
        return reject("缺少已确认标定的位置范围")
    if (audit.get("status") != "AMBIGUOUS" or audit.get("searchTruncated") is not False
            or audit.get("distinctHypothesisCount") != 2 or audit.get("runnerUpOverlapIoU") != 0
            or not isinstance(audit.get("runnerUp"),dict)):
        return reject("位置约束需要完整、互不重叠的两个候选")
    try:
        height,width = image_shape[:2]
        radius_um = float(calibration_location["maximumCenterErrorUm"])
        target = np.asarray(calibration_location["targetStageUm"],float)
        actual = np.asarray(stage_center_um,float)
        axes = np.asarray(pixel_axis_matrix,float)
        if (target.shape != (2,) or actual.shape != (2,) or axes.shape != (2,2)
                or not np.isfinite([target,actual]).all() or not np.isfinite(axes).all()
                or not math.isfinite(pixel_size_um) or pixel_size_um <= 0
                or not math.isfinite(field_rotation_deg) or not math.isfinite(radius_um)
                or not 0 < radius_um <= min(height,width)*pixel_size_um*.2
                or not _valid_axes(axes)):
            return reject("标定位置、像素轴或误差范围无效")
        if np.linalg.norm(target-actual) > 2.:
            return reject("实际采图台位置偏离预测位置，不能用位置约束放行")
        angle=math.radians(field_rotation_deg)
        rotation=np.asarray([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        predicted=np.asarray([width/2,height/2])+np.linalg.solve(rotation@axes,target-actual)/pixel_size_um
        radius=radius_um/(pixel_size_um*np.linalg.svd(axes,compute_uv=False).min())
        if np.any(predicted-radius<0) or np.any(predicted+radius>=np.asarray([width,height])):
            return reject("标定搜索范围不能完整放入视场")
        centers=[np.asarray(a["centerPixel"],float) for a in [audit,audit["runnerUp"]]]
        if any(c.shape!=(2,) or not np.isfinite(c).all() or np.any(c<0)
               or np.any(c>=np.asarray([width,height])) for c in centers):
            return reject("模板候选坐标无效")
        from .particle_refinement import RefinementSettings
        score_floor = RefinementSettings().match_min_score if minimum_template_score is None else float(minimum_template_score)
        if not math.isfinite(score_floor) or not 0 < score_floor < 1:
            return reject("模板相关分数阈值无效")
        if any(not math.isfinite(float(a['score'])) or float(a['score'])<score_floor
               for a in [audit,audit['runnerUp']]):
            return reject("模板相关分数不足")
        extent_um=particle.length_um/2
        if not math.isfinite(extent_um) or extent_um<=0:
            return reject("颗粒尺寸无效")
        distances=[float(np.linalg.norm(rotation@axes@(c-predicted)))*pixel_size_um for c in centers]
        result.update(predictedCenterPx=predicted.tolist(),maximumCenterErrorUm=radius_um,
                      candidateDistancesUm=distances,particleHalfLengthUm=extent_um,
                      source="task_calibration_acceptance_maxErrorUm")
        if distances[0]+extent_um >= radius_um:
            return reject("原第一候选不能完整位于已确认标定范围内")
        if distances[1]-extent_um <= radius_um:
            return reject("竞争颗粒仍与标定允许范围相交，保留歧义拒绝")
        return {**result,"accepted":True,
                "reason":"原第一候选位于标定允许范围内，竞争颗粒在范围外；仍须最终实测轮廓复核"}
    except (ValueError,TypeError,KeyError,np.linalg.LinAlgError):
        return reject("标定位置约束信息不完整或无效")
