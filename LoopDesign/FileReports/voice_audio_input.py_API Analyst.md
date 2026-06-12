# API Analyst Report: voice\audio_input.py

## Dependencies
- `from __future__ import annotations`
- `from dataclasses import dataclass`
- `from textwrap import dedent`
- `from typing import TYPE_CHECKING`
- `from typing import TypeAlias`
- `from typing import cast`
- `from streamlit.elements.lib.file_uploader_utils import enforce_filename_restriction`
- `from streamlit.elements.lib.form_utils import current_form_id`
- `from streamlit.elements.lib.layout_utils import LayoutConfig`
- `from streamlit.elements.lib.layout_utils import validate_width`
- `from streamlit.elements.lib.policies import check_widget_policies`
- `from streamlit.elements.lib.policies import maybe_raise_label_warnings`
- `from streamlit.elements.lib.utils import Key`
- `from streamlit.elements.lib.utils import LabelVisibility`
- `from streamlit.elements.lib.utils import compute_and_register_element_id`
- `from streamlit.elements.lib.utils import get_label_visibility_proto_value`
- `from streamlit.elements.lib.utils import to_key`
- `from streamlit.elements.widgets.file_uploader import _get_upload_files`
- `from streamlit.errors import StreamlitAPIException`
- `from streamlit.proto.AudioInput_pb2 import AudioInput as AudioInputProto`
- `from streamlit.proto.Common_pb2 import FileUploaderState as FileUploaderStateProto`
- `from streamlit.proto.Common_pb2 import UploadedFileInfo as UploadedFileInfoProto`
- `from streamlit.runtime.metrics_util import gather_metrics`
- `from streamlit.runtime.scriptrunner import ScriptRunContext`
- `from streamlit.runtime.scriptrunner import get_script_run_ctx`
- `from streamlit.runtime.state import WidgetArgs`
- `from streamlit.runtime.state import WidgetCallback`
- `from streamlit.runtime.state import WidgetKwargs`
- `from streamlit.runtime.state import register_widget`
- `from streamlit.runtime.uploaded_file_manager import DeletedFile`
- `from streamlit.runtime.uploaded_file_manager import UploadedFile`

## Configuration Variables
- `ALLOWED_SAMPLE_RATES` = `{8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000}`

## Schemas & API Contracts (Classes)

### Class `AudioInputSerde`
**Methods:**
- `def serialize(self, audio_file: SomeUploadedAudioFile) -> FileUploaderStateProto`
- `def deserialize(self, ui_value: FileUploaderStateProto | None) -> SomeUploadedAudioFile`


### Class `AudioInputMixin`
**Methods:**
- @gather_metrics('audio_input')
- `def audio_input(self, label: str, *, sample_rate: int | None=16000, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, disabled: bool=False, label_visibility: LabelVisibility='visible', width: WidthWithoutContent='stretch') -> UploadedFile | None`
  - *Display a widget that returns an audio recording from the user's microphone.*
- `def _audio_input(self, label: str, sample_rate: int | None=16000, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, disabled: bool=False, label_visibility: LabelVisibility='visible', width: WidthWithoutContent='stretch', ctx: ScriptRunContext | None=None) -> UploadedFile | None`
- @property
- `def dg(self) -> DeltaGenerator`
  - *Get our DeltaGenerator.*

