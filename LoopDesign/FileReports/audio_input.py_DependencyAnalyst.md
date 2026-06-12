# Dependency Analysis Report for voice\audio_input.py

## Library Requirements
- from __future__ import annotations
- from dataclasses import dataclass
- from streamlit.delta_generator import DeltaGenerator
- from streamlit.elements.lib.file_uploader_utils import enforce_filename_restriction
- from streamlit.elements.lib.form_utils import current_form_id
- from streamlit.elements.lib.layout_utils import LayoutConfig
- from streamlit.elements.lib.layout_utils import WidthWithoutContent
- from streamlit.elements.lib.layout_utils import validate_width
- from streamlit.elements.lib.policies import check_widget_policies
- from streamlit.elements.lib.policies import maybe_raise_label_warnings
- from streamlit.elements.lib.utils import Key
- from streamlit.elements.lib.utils import LabelVisibility
- from streamlit.elements.lib.utils import compute_and_register_element_id
- from streamlit.elements.lib.utils import get_label_visibility_proto_value
- from streamlit.elements.lib.utils import to_key
- from streamlit.elements.widgets.file_uploader import _get_upload_files
- from streamlit.errors import StreamlitAPIException
- from streamlit.proto.AudioInput_pb2 import AudioInput
- from streamlit.proto.Common_pb2 import FileUploaderState
- from streamlit.proto.Common_pb2 import UploadedFileInfo
- from streamlit.runtime.metrics_util import gather_metrics
- from streamlit.runtime.scriptrunner import ScriptRunContext
- from streamlit.runtime.scriptrunner import get_script_run_ctx
- from streamlit.runtime.state import WidgetArgs
- from streamlit.runtime.state import WidgetCallback
- from streamlit.runtime.state import WidgetKwargs
- from streamlit.runtime.state import register_widget
- from streamlit.runtime.uploaded_file_manager import DeletedFile
- from streamlit.runtime.uploaded_file_manager import UploadedFile
- from textwrap import dedent
- from typing import TYPE_CHECKING
- from typing import TypeAlias
- from typing import cast

## Service Dependencies
- URL: http://www.apache.org/licenses/LICENSE-2.0
- URL: https://doc-audio-input-high-rate.streamlit.app/
- URL: https://doc-audio-input.streamlit.app/
- URL: https://docs.streamlit.io/develop/api-reference/text/st.markdown

## Hidden Execution Links
- None detected

## Configurations / Assumptions
- Analyzed via AST and regex.
