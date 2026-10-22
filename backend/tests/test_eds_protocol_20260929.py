from google.protobuf import descriptor_pb2

from backend.cleanliness_core.eds.generated.v2 import job_events_pb2 as events, jobs_pb2 as jobs


def test_old_completed_wire_and_original_field_numbers_remain_compatible():
    # Old wire: result_text="ok", accumulated_counts=128; no new fields present.
    old = events.JobCompleted.FromString(b'\x0a\x02ok\x40\x80\x01')
    assert old.result_text == 'ok' and old.accumulated_counts == 128
    assert not old.HasField('vda_classification')
    assert not old.element_quantifications
    assert old.stop_reason == events.SPECTRUM_STOP_REASON_UNSPECIFIED
    expected = {'result_text':1,'maps':2,'index_concentrations':3,'quantify_result_map':4,
                'material_classification':5,'spectrum_image':6,'elapsed_seconds':7,'accumulated_counts':8}
    assert {k:events.JobCompleted.DESCRIPTOR.fields_by_name[k].number for k in expected} == expected


def test_optional_sigma_distinguishes_absent_from_zero():
    q = events.ElementQuantification(element='Cu', weight_percent=4.82)
    assert not q.HasField('weight_percent_sigma')
    q.weight_percent_sigma = 0
    parsed = events.ElementQuantification.FromString(q.SerializeToString())
    assert parsed.HasField('weight_percent_sigma') and parsed.weight_percent_sigma == 0


def test_new_completed_fields_round_trip_without_losing_two_classifications():
    completed = events.JobCompleted(
        quantify_result_map={'C':78.25,'Cu':4.82,'O':14.74,'Zn':2.19},
        element_quantifications=[events.ElementQuantification(element='Cu',weight_percent=4.82,weight_percent_sigma=.3)],
        material_classification=events.MaterialClassification(material_class='Si/Si-C/Si-N',hardness_label='HV>1000',
            warnings=['Matched a catch-all class with no child criteria.'],boundary_notes=['Copper: near threshold']),
        vda_classification=events.MaterialClassification(material_class='brass',hardness_label='V1',confidence=1),
        oxford_weight_percent={'C':78.25,'Cu':4.82,'O':14.74,'Zn':2.19},
        vda_analysis_composition={'Cu':68.73,'Zn':31.27,'O':0},
        real_time_seconds=2.405,live_time_seconds=2.143,elapsed_seconds=2.85,
        target_counts=50000,accumulated_counts=66812,stop_reason=events.SPECTRUM_STOP_REASON_TARGET_REACHED,
        classification_report='server report',started_at='2026-09-29 14:34:04',ended_at='2026-09-29 14:34:07',quant_method='default')
    event = events.JobEvent(seq=14,completed=completed)
    parsed = events.JobEvent.FromString(event.SerializeToString())
    assert parsed == event
    assert parsed.WhichOneof('payload') == 'completed'
    assert 'C' not in parsed.completed.vda_analysis_composition
    assert 'O' in parsed.completed.vda_analysis_composition
    assert parsed.completed.vda_analysis_composition['O'] == 0


def test_fast_is_reserved_and_existing_stop_conditions_preserved():
    descriptor=descriptor_pb2.FileDescriptorProto.FromString(jobs.DESCRIPTOR.serialized_pb)
    preset=next(e for e in descriptor.enum_type if e.name=='SpectrumAcquisitionPreset')
    assert 'SPECTRUM_ACQUISITION_PRESET_FAST' in preset.reserved_name
    assert any(r.start<=1<=r.end for r in preset.reserved_range)
    assert not hasattr(jobs,'SPECTRUM_ACQUISITION_PRESET_FAST')
    req=jobs.StartSpectrumAcquisitionJobRequest(target_counts=50000,max_duration_seconds=600)
    assert req.WhichOneof('stop_condition')=='target_counts'
    req.preset=jobs.SPECTRUM_ACQUISITION_PRESET_AUTOMATIC
    assert req.WhichOneof('stop_condition')=='preset' and req.HasField('preset')
    assert req.target_counts==0
