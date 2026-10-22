from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "sample_cleanliness_report"


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        os.environ["ZEM_TASK_ROOT"] = str(Path(cls.temporary.name) / "tasks")
        from backend import server
        cls.server = server
        # Other tests may import server first. Isolate mutable module paths too,
        # not just the environment evaluated at module import time.
        cls.path_patch = mock.patch.multiple(
            server,
            TASK_ROOT=Path(cls.temporary.name) / "tasks",
            SETTINGS_PATH=Path(cls.temporary.name) / "settings.json",
            ENGINEER_OVERRIDES_PATH=Path(cls.temporary.name) / ".engineer-settings.json",
            LOG_ROOT=Path(cls.temporary.name) / "logs",
            LOG_FILE=Path(cls.temporary.name) / "logs" / "backend.jsonl",
            LOG_ARCHIVE=Path(cls.temporary.name) / "logs" / "backend.previous.jsonl",
        )
        cls.path_patch.start()
        server.TASK_ROOT.mkdir(parents=True, exist_ok=True)
        server.LOG_ROOT.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.path_patch.stop()
        cls.temporary.cleanup()

    def test_packaged_models_are_present_and_nonempty(self) -> None:
        rows = self.server.model_status()
        self.assertEqual({row["id"] for row in rows}, {"particle", "fixture", "hole"})
        self.assertTrue(all(row["ready"] for row in rows))
        self.assertTrue(all(row["size"] > 1_000_000 for row in rows))
        self.assertTrue(all(len(row["sha256"]) == 64 for row in rows))

    def test_real_optical_export_preview_and_filter(self) -> None:
        source = FIXTURE_ROOT / "optical.json"
        result = self.server._load_optical(source, {"minLengthUm": 70, "maxLengthUm": 72})
        self.assertEqual(result["particleCount"], 5)
        self.assertEqual(result["selectedCount"], 2)
        self.assertEqual({row["sequence"] for row in result["particles"]}, {3, 4})

    def test_selected_preview_particles_respects_requested_subset(self) -> None:
        preview = {"particles": [{"sequence": 3}, {"sequence": 4}, {"sequence": 8}]}
        selected = self.server._selected_preview_particles(preview, [8, 3])
        self.assertEqual([row["sequence"] for row in selected], [8, 3])

    def test_selected_preview_particles_rejects_filtered_out_sequence(self) -> None:
        preview = {"particles": [{"sequence": 3}, {"sequence": 4}]}
        with self.assertRaisesRegex(ValueError, "不在当前筛选结果"):
            self.server._selected_preview_particles(preview, [3, 9])

    def test_revise_task_input_keeps_task_id_and_archives_invalidated_calibration(self) -> None:
        source = FIXTURE_ROOT / "optical.json"
        preview = self.server._load_optical(source, {"minLengthUm": 70, "maxLengthUm": 72})
        task_id = "revise-input-task"
        task_dir = self.server.TASK_ROOT / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        body = {
            "inputPath": str(source),
            "filters": {"minLengthUm": 70, "maxLengthUm": 72},
            "selectedSequences": [row["sequence"] for row in preview["particles"]],
            "mode": "multi_particle",
        }
        draft, _ = self.server._build_task_draft(task_id, body)
        self.server.atomic_json(task_dir / "draft.json", draft)
        formal = json.loads(json.dumps(draft))
        formal.update(status="CALIBRATION_ACCEPTED", calibration={"strategy": "multi_particle", "review": {"accepted": True}})
        self.server.atomic_json(task_dir / "task.json", formal)

        unchanged = self.server.revise_task_input(task_id, body)
        self.assertFalse(unchanged["inputRevised"])
        self.assertTrue((task_dir / "task.json").is_file())

        revised_body = {**body, "mode": "ab_points"}
        with self.assertRaisesRegex(ValueError, "标定结果失效"):
            self.server.revise_task_input(task_id, revised_body)

        revised = self.server.revise_task_input(task_id, {**revised_body, "confirmReset": True})
        self.assertEqual(revised["taskId"], task_id)
        self.assertEqual(revised["status"], "DRAFT")
        self.assertEqual(revised["revision"], 1)
        self.assertEqual(revised["request"]["mode"], "ab_points")
        self.assertFalse((task_dir / "task.json").exists())
        self.assertTrue((task_dir / "draft.json").is_file())
        self.assertEqual(len(list((task_dir / "revisions").glob("task-before-input-revision-*.json"))), 1)

    def test_atomic_json_round_trip(self) -> None:
        destination = Path(self.temporary.name) / "task" / "draft.json"
        payload = {"taskId": "sample", "status": "DRAFT", "name": "正极极片"}
        self.server.atomic_json(destination, payload)
        self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), payload)

    def test_atomic_json_uses_shared_windows_safe_writer(self) -> None:
        destination = Path(self.temporary.name) / "task" / "task.json"
        payload = {"taskId": "sample", "status": "RUNNING"}

        with mock.patch.object(self.server, "write_json_atomic") as writer:
            self.server.atomic_json(destination, payload)

        writer.assert_called_once_with(destination, payload)

    def test_structured_log_is_written_and_sensitive_fields_are_redacted(self) -> None:
        log_file = Path(self.temporary.name) / "logs" / "backend.jsonl"
        log_archive = log_file.with_name("backend.previous.jsonl")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with mock.patch.object(self.server, "LOG_FILE", log_file), mock.patch.object(self.server, "LOG_ARCHIVE", log_archive):
            self.server.log_event("info", "calibration_stage_finished", taskId="task-1", stage="capture", elapsedSeconds=1.25, password="do-not-store", nested={"token": "also-secret", "value": 7})

        record = json.loads(log_file.read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(record["event"], "calibration_stage_finished")
        self.assertEqual(record["taskId"], "task-1")
        self.assertEqual(record["stage"], "capture")
        self.assertEqual(record["elapsedSeconds"], 1.25)
        self.assertIsNotNone(datetime.fromisoformat(record["timestamp"]).tzinfo)
        self.assertEqual(record["password"], "[redacted]")
        self.assertEqual(record["nested"]["token"], "[redacted]")
        self.assertEqual(record["nested"]["value"], 7)

    def test_retry_ab_refinement_only_resets_task_state(self) -> None:
        task_id = "ab-review-task"
        task_dir = self.server.TASK_ROOT / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "taskId": task_id,
            "status": "CALIBRATION_PENDING_REVIEW",
            "calibration": {
                "strategy": "ab_points",
                "result": {"verified": True},
                "refinedTransform": {"verified": True},
                "review": None,
                "abPointResult": {
                    "status": "SEM_REFINED",
                    "semRefinement": {
                        "status": "SUCCEEDED",
                        "validation": [{"sequence": 12}],
                    },
                },
            },
            "error": None,
        }
        self.server.atomic_json(task_dir / "task.json", manifest)

        reset = self.server.retry_ab_refinement(task_id)

        self.assertEqual(reset["status"], "AB_POINTS_PENDING_SEM_REFINEMENT")
        ab_result = reset["calibration"]["abPointResult"]
        self.assertEqual(ab_result["status"], "PENDING_SEM_REFINEMENT")
        self.assertNotIn("semRefinement", ab_result)
        self.assertEqual(ab_result["semRefinementHistory"][0]["status"], "SUCCEEDED")
        self.assertIsNone(reset["calibration"]["result"])
        self.assertIsNone(reset["calibration"]["refinedTransform"])

    def test_acceptance_mock_spectrum_handles_tuple_contour(self) -> None:
        particle = types.SimpleNamespace(sequence=7, area_um2=123.4)
        candidate = types.SimpleNamespace(
            candidate=types.SimpleNamespace(contour=((1, 2), (3, 4), (5, 6)))
        )

        result = self.server.AcceptanceMockSpectrumAnalyzer().analyze(
            particle,
            candidate,
            Path(self.temporary.name) / "sem.png",
            (25, 31),
        )

        self.assertTrue(result["mock"])
        self.assertEqual(result["device"], "Mock EDS (acceptance test)")
        self.assertTrue(result["elements"])
        self.assertEqual(result["target"]["x"], 25)
        self.assertEqual(result["target"]["y"], 31)

    def test_reset_analysis_archives_summary_without_hardware_action(self) -> None:
        task_id = "stale-analysis-task"
        task_dir = self.server.TASK_ROOT / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "taskId": task_id,
            "status": "RUNNING",
            "calibration": {"review": {"accepted": True}},
            "particles": [
                {"sequence": 1, "status": "SUCCEEDED"},
                {"sequence": 2, "status": "FAILED"},
            ],
            "reports": None,
            "cleanup": {"stageMoved": True},
            "analysisStartedAt": "2026-08-13T18:00:00+08:00",
            "executionOrder": [1, 2],
            "executionRoute": {"algorithm": "old", "optimizedDistanceUm": 10},
            "baselineImaging": {"shape": [512, 512]},
            "error": "old error",
        }
        self.server.atomic_json(task_dir / "task.json", manifest)

        reset = self.server.reset_analysis(task_id)

        self.assertEqual(reset["status"], "CALIBRATION_ACCEPTED")
        self.assertEqual(reset["particles"], [])
        self.assertIsNone(reset["reports"])
        self.assertEqual(reset["cleanup"], {})
        self.assertNotIn("executionOrder", reset)
        self.assertNotIn("executionRoute", reset)
        self.assertNotIn("baselineImaging", reset)
        self.assertEqual(reset["analysisHistory"][0]["processed"], 2)
        self.assertEqual(reset["analysisHistory"][0]["succeeded"], 1)
        self.assertEqual(reset["analysisHistory"][0]["failed"], 1)

    def test_analysis_resume_state_counts_terminal_records_and_next_rank(self) -> None:
        manifest = {
            "status": "FAILED",
            "calibration": {"review": {"accepted": True}},
            "selection": {"sequences": [50, 348, 312, 356]},
            "executionOrder": [50, 312, 348, 356],
            "particles": [
                {"rank": 1, "sequence": 50, "status": "SUCCEEDED"},
                {"rank": 2, "sequence": 312, "status": "FAILED"},
                {"rank": 3, "sequence": 348, "status": "RUNNING"},
            ],
        }

        state = self.server.analysis_resume_state(manifest)

        self.assertEqual(state["processed"], 2)
        self.assertEqual(state["total"], 4)
        self.assertEqual(state["remaining"], 2)
        self.assertEqual(state["nextRank"], 3)
        self.assertEqual(state["nextSequence"], 348)

    def test_analysis_resume_state_rejects_fully_processed_task(self) -> None:
        manifest = {
            "status": "FAILED",
            "calibration": {"review": {"accepted": True}},
            "selection": {"sequences": [1, 2]},
            "executionOrder": [2, 1],
            "particles": [
                {"rank": 1, "sequence": 2, "status": "SUCCEEDED"},
                {"rank": 2, "sequence": 1, "status": "FAILED"},
            ],
        }

        with self.assertRaisesRegex(ValueError, "没有未处理颗粒"):
            self.server.analysis_resume_state(manifest)

    def test_desktop_task_id_is_stable_when_ab_assignment_materializes_manifest(self) -> None:
        task_id = "desktop-folder-id"
        task_dir = self.server.TASK_ROOT / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        draft = {
            "taskId": task_id,
            "input": {"path": "optical.json"},
            "request": {"mode": "ab_points"},
        }
        self.server.atomic_json(task_dir / "draft.json", draft)
        self.server.AB_SESSIONS[task_id] = {
            "inputSha256": "sha",
            "image": object(),
            "bounds": (0, 0, 1, 1),
            "detection": object(),
        }
        legacy_result = {"taskId": "legacy-random-id", "status": "AB_POINTS_PENDING_SEM_REFINEMENT"}
        with mock.patch.object(self.server, "_make_config", return_value=object()), mock.patch(
            "backend.cleanliness_core.cleanliness_task.save_ab_point_calibration",
            return_value=legacy_result,
        ):
            result = self.server.save_ab_assignment(task_id, {"aIndex": 0, "bIndex": 1})

        persisted = self.server.read_json(task_dir / "task.json")
        self.assertEqual(result["taskId"], task_id)
        self.assertEqual(persisted["taskId"], task_id)

    def test_microscope_status_uses_udp_validator_without_tcp_preflight(self) -> None:
        with mock.patch(
            "backend.cleanliness_core.eds.microscope_jsonrpc.probe_microscope_status",
            return_value="Idle",
        ) as validate, mock.patch(
            "socket.create_connection",
            side_effect=AssertionError("microscope status must not use TCP"),
        ):
            result = self.server._microscope_status("192.168.0.15", 61557)
        self.assertEqual(result, {"status": "connected", "detail": "Idle"})
        validate.assert_called_once_with(
            "192.168.0.15",
            61557,
            timeout=1.0,
        )

    def test_engineer_unlock_rejects_wrong_password(self) -> None:
        with self.assertRaises(PermissionError):
            self.server.unlock_engineer("wrong-password")

    def test_preflight_blocks_every_hardware_job_before_core_work(self):
        for operation in ('calibration','ab_refine','analysis','analysis_resume'):
            with self.subTest(operation=operation), mock.patch.object(self.server,'JOBS',self.server.JobManager()):
                task_id='blocked-'+operation
                folder=self.server.TASK_ROOT/task_id;folder.mkdir(exist_ok=True)
                original={'taskId':task_id,'status':'CALIBRATION_ACCEPTED','input':{'path':str(FIXTURE_ROOT/'optical.json')},'request':{}}
                self.server.atomic_json(folder/'task.json',original)
                job=self.server.JOBS.create(task_id,operation)
                check={'enabled':True,'passed':False,'items':[{'id':'gun','label':'电子枪','status':'FAIL','detail':'未开启'}]}
                with mock.patch.object(self.server,'_make_config',return_value=object()), \
                     mock.patch.object(self.server,'_execute_preflight',return_value=check), \
                     mock.patch('backend.cleanliness_core.cleanliness_task.run_cleanliness_analysis') as analysis, \
                     mock.patch('backend.cleanliness_core.cleanliness_task.run_cleanliness_calibration') as calibration, \
                     mock.patch('backend.cleanliness_core.cleanliness_task.refine_ab_point_calibration') as refine:
                    self.server._run_job_impl(job['id'],task_id,operation)
                    analysis.assert_not_called();calibration.assert_not_called();refine.assert_not_called()
                saved=self.server.read_json(folder/'task.json')
                self.assertEqual(saved['status'],original['status'])
                self.assertFalse(saved['preflight']['passed'])
                self.assertEqual(self.server.JOBS.get(job['id'])['status'],'failed')

    def test_preflight_disabled_does_not_read_task_or_device(self):
        with mock.patch.object(self.server,'app_settings',return_value={'preflightEnabled':False}), \
             mock.patch('backend.cleanliness_core.preflight.ZemPreflightReader') as reader:
            result=self.server._execute_preflight(None,None,'analysis',should_stop=lambda:False,progress=lambda *args:None)
            self.assertFalse(result['enabled']);self.assertTrue(result['passed'])
            reader.assert_not_called()

    def test_shutdown_and_report_settings_migrate_and_freeze_per_task(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(self.server, 'SETTINGS_PATH', Path(folder)/'settings.json'):
            root = Path(folder)
            self.server.atomic_json(self.server.SETTINGS_PATH, {'gunOffAfterTask': False})
            self.assertTrue(self.server.app_settings()['gunOffAfterTask'])  # legacy forced-off value
            self.assertTrue(self.server.app_settings()['showReportCoordinates'])
            self.assertTrue(self.server.app_settings()['preflightEnabled'])
            self.assertEqual(self.server.app_settings()['captureLevel'], 7)
            self.assertEqual(self.server.app_settings()['candidateScoreMargin'], 0.07)
            self.assertEqual(self.server.app_settings()['templateMatchMinScore'], 0.65)
            self.assertEqual(self.server.app_settings()['contourOverlapMin'], 0.5)
            self.assertEqual(self.server.app_settings()['maxCaptureAttempts'], 2)
            self.server.atomic_json(self.server.SETTINGS_PATH, {
                'settingsSchemaVersion': 5, 'templateMatchMinScore': 0.62,
                'contourOverlapMin': 0.2, 'maxCaptureAttempts': 4,
            })
            migrated = self.server.app_settings()
            self.assertEqual(migrated['templateMatchMinScore'], 0.62)
            self.assertEqual(migrated['contourOverlapMin'], 0.5)
            self.assertEqual(migrated['maxCaptureAttempts'], 2)
            self.server.save_app_settings({'gunOffAfterTask': False, 'showReportCoordinates': False, 'preflightEnabled':False, 'captureLevel':4, 'candidateScoreMargin':0.05, 'templateMatchMinScore':0.55, 'contourOverlapMin':0.4, 'maxCaptureAttempts':3})
            saved = self.server.app_settings()
            self.assertFalse(saved['gunOffAfterTask'])
            self.assertFalse(saved['showReportCoordinates'])
            self.assertFalse(saved['preflightEnabled'])
            self.assertEqual(saved['captureLevel'], 4)
            self.assertEqual(saved['candidateScoreMargin'], 0.05)
            self.assertEqual(saved['templateMatchMinScore'], 0.55)
            self.assertEqual(saved['contourOverlapMin'], 0.4)
            self.assertEqual(saved['maxCaptureAttempts'], 3)
            source = FIXTURE_ROOT/'optical.json'
            draft, _ = self.server._build_task_draft('options', {'inputPath': str(source), 'mode': 'ab_manual'})
            request = draft['request']
            self.assertFalse(request['gunOffAfterTask'])
            self.assertEqual(request['captureLevel'], 4)
            self.assertEqual(request['candidateScoreMargin'], 0.05)
            self.assertEqual(request['templateMatchMinScore'], 0.55)
            self.assertEqual(request['contourOverlapMin'], 0.4)
            self.assertEqual(request['maxCaptureAttempts'], 3)
            old_dir = root/'old'; old_dir.mkdir()
            old = self.server._make_config(source, old_dir, request, for_analysis=True)
            self.assertFalse(old.gun_off_after_task)
            self.assertEqual(old.pixel_avg, 4)
            self.assertFalse(self.server.read_json(old.settings_path)['report']['showCoordinates'])
            old_snapshot = self.server.read_json(old.settings_path)
            self.assertEqual(old_snapshot['task']['captureLevel'], 4)
            self.assertEqual(old_snapshot['abPoint']['semRefine']['pixelAvg'], 4)
            self.assertEqual(old_snapshot['abParticle']['pixel_avg'], 4)
            self.assertEqual(old_snapshot['abParticle']['match_min_margin'], 0.05)
            self.assertEqual(old_snapshot['abParticle']['match_min_score'], 0.55)
            self.assertEqual(old_snapshot['task']['candidateScoreMargin'], 0.05)
            self.assertEqual(old_snapshot['task']['templateMatchMinScore'], 0.55)
            self.assertEqual(old_snapshot['task']['contourOverlapMin'], 0.4)
            self.assertEqual(old_snapshot['task']['maxCaptureAttempts'], 3)
            self.assertEqual(old.max_capture_attempts, 3)
            self.server.save_app_settings({'gunOffAfterTask': True, 'showReportCoordinates': True, 'captureLevel':6, 'candidateScoreMargin':0.09, 'templateMatchMinScore':0.7, 'contourOverlapMin':0.8, 'maxCaptureAttempts':1})
            self.assertFalse(self.server._make_config(source, old_dir, request, for_analysis=True).gun_off_after_task)
            self.assertEqual(self.server._make_config(source, old_dir, request, for_analysis=True).pixel_avg, 4)
            self.assertEqual(self.server._make_config(source, old_dir, request, for_analysis=True).max_capture_attempts, 3)
            self.assertEqual(self.server.read_json(old.settings_path)['abParticle']['match_min_margin'], 0.05)
            self.assertEqual(self.server.read_json(old.settings_path)['abParticle']['match_min_score'], 0.55)
            self.assertEqual(self.server.read_json(old.settings_path)['task']['contourOverlapMin'], 0.4)
            self.assertEqual(self.server.read_json(old.settings_path)['task']['maxCaptureAttempts'], 3)
            new_dir = root/'new'; new_dir.mkdir()
            calibration = self.server._make_config(source, new_dir, {}, for_analysis=False)
            analysis = self.server._make_config(source, new_dir, {}, for_analysis=True)
            self.assertFalse(calibration.gun_off_after_task)
            self.assertTrue(analysis.gun_off_after_task)
            self.assertEqual(analysis.pixel_avg, 6)
            new_snapshot = self.server.read_json(analysis.settings_path)
            self.assertEqual(new_snapshot['abPoint']['semRefine']['pixelAvg'], 6)
            self.assertEqual(new_snapshot['abParticle']['pixel_avg'], 6)
            self.assertEqual(new_snapshot['abParticle']['match_min_margin'], 0.09)
            self.assertEqual(new_snapshot['abParticle']['match_min_score'], 0.7)
            self.assertEqual(new_snapshot['task']['candidateScoreMargin'], 0.09)
            self.assertEqual(new_snapshot['task']['templateMatchMinScore'], 0.7)
            self.assertEqual(new_snapshot['task']['contourOverlapMin'], 0.8)
            self.assertEqual(new_snapshot['task']['maxCaptureAttempts'], 1)
            self.assertEqual(analysis.max_capture_attempts, 1)
            self.assertTrue(self.server.read_json(analysis.settings_path)['report']['showCoordinates'])
            with self.assertRaises(ValueError):
                self.server.save_app_settings({'gunOffAfterTask': 'false'})
            for invalid in (0, 8, True, '7'):
                with self.assertRaises(ValueError):
                    self.server.save_app_settings({'captureLevel': invalid})
                with self.assertRaises(ValueError):
                    self.server._build_task_draft('invalid-capture-level', {
                        'inputPath': str(source), 'captureLevel': invalid,
                    })
            for invalid in (0, 1, -0.1, True, '0.07', float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    self.server.save_app_settings({'candidateScoreMargin': invalid})
                with self.assertRaises(ValueError):
                    self.server._build_task_draft('invalid-candidate-margin', {
                        'inputPath': str(source), 'candidateScoreMargin': invalid,
                    })
                with self.assertRaises(ValueError):
                    self.server.save_app_settings({'templateMatchMinScore': invalid})
                with self.assertRaises(ValueError):
                    self.server._build_task_draft('invalid-template-score', {
                        'inputPath': str(source), 'templateMatchMinScore': invalid,
                    })
                with self.assertRaises(ValueError):
                    self.server.save_app_settings({'contourOverlapMin': invalid})
                with self.assertRaises(ValueError):
                    self.server._build_task_draft('invalid-contour-overlap', {
                        'inputPath': str(source), 'contourOverlapMin': invalid,
                    })
            for invalid in (0, 6, True, '2', 2.5):
                with self.assertRaises(ValueError):
                    self.server.save_app_settings({'maxCaptureAttempts': invalid})
                with self.assertRaises(ValueError):
                    self.server._build_task_draft('invalid-capture-count', {
                        'inputPath': str(source), 'maxCaptureAttempts': invalid,
                    })

    def test_engineer_settings_validate_persist_and_snapshot(self) -> None:
        unlocked = self.server.unlock_engineer("zeptools")
        token = unlocked["token"]
        loaded = self.server.engineer_settings(token)
        self.assertEqual(len(loaded["definitions"]), 15)
        self.assertNotIn("task.gunOffAfterTask", loaded["values"])

        values = dict(loaded["values"])
        values["abPoint.fixtureConfidence"] = 0.2
        saved = self.server.save_engineer_settings(token, values)
        self.assertEqual(saved["values"]["abPoint.fixtureConfidence"], 0.2)

        snapshot_dir = Path(self.temporary.name) / "snapshot"
        snapshot_dir.mkdir()
        snapshot = self.server.read_json(self.server._task_config_path(snapshot_dir))
        self.assertEqual(snapshot["abPoint"]["fixtureConfidence"], 0.2)
        self.assertFalse(snapshot["task"]["gunOnAtStart"])
        self.assertTrue(snapshot["task"]["gunOffAfterTask"])
        self.assertTrue(Path(snapshot["experienceCalibration"]).is_absolute())

        request = self.server.app_settings()
        request["microscopeHost"] = "old-device"
        config = self.server._make_config(
            FIXTURE_ROOT / "optical.json",
            snapshot_dir,
            request,
        )
        self.assertEqual(config.microscope_host, "old-device")

        values["abPoint.fixtureConfidence"] = 0.3
        self.server.save_engineer_settings(token, values)
        unchanged = self.server.read_json(self.server._task_config_path(snapshot_dir))
        self.assertEqual(unchanged["abPoint"]["fixtureConfidence"], 0.2)

        values["calibration.acceptance.meanErrorUm"] = 200
        values["calibration.acceptance.maxErrorUm"] = 100
        with self.assertRaisesRegex(ValueError, "最大误差上限"):
            self.server.save_engineer_settings(token, values)

        self.server.lock_engineer(token)
        with self.assertRaises(PermissionError):
            self.server.engineer_settings(token)


if __name__ == "__main__":
    unittest.main()
