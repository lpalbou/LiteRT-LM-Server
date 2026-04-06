#![cfg(feature = "litert")]

use std::os::raw::{c_char, c_int, c_void};

#[repr(C)]
pub struct LiteRtLmEngine {
    _private: [u8; 0],
}

#[repr(C)]
pub struct LiteRtLmEngineSettings {
    _private: [u8; 0],
}

#[repr(C)]
pub struct LiteRtLmSessionConfig {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum LiteRtLmSamplerType {
    TypeUnspecified = 0,
    TopK = 1,
    TopP = 2,
    Greedy = 3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LiteRtLmSamplerParams {
    pub r#type: LiteRtLmSamplerType,
    pub top_k: i32,
    pub top_p: f32,
    pub temperature: f32,
    pub seed: i32,
}

#[repr(C)]
pub struct LiteRtLmConversationConfig {
    _private: [u8; 0],
}

#[repr(C)]
pub struct LiteRtLmConversation {
    _private: [u8; 0],
}

#[repr(C)]
pub struct LiteRtLmJsonResponse {
    _private: [u8; 0],
}

pub type LiteRtLmStreamCallback = Option<
    unsafe extern "C" fn(
        callback_data: *mut c_void,
        chunk: *const c_char,
        is_final: bool,
        error_msg: *const c_char,
    ),
>;

unsafe extern "C" {
    pub fn litert_lm_set_min_log_level(level: c_int);

    pub fn litert_lm_session_config_create() -> *mut LiteRtLmSessionConfig;

    pub fn litert_lm_session_config_set_max_output_tokens(
        config: *mut LiteRtLmSessionConfig,
        max_output_tokens: c_int,
    );

    pub fn litert_lm_session_config_set_sampler_params(
        config: *mut LiteRtLmSessionConfig,
        sampler_params: *const LiteRtLmSamplerParams,
    );

    pub fn litert_lm_session_config_delete(config: *mut LiteRtLmSessionConfig);

    pub fn litert_lm_engine_settings_create(
        model_path: *const c_char,
        backend_str: *const c_char,
        vision_backend_str: *const c_char,
        audio_backend_str: *const c_char,
    ) -> *mut LiteRtLmEngineSettings;

    pub fn litert_lm_engine_settings_delete(settings: *mut LiteRtLmEngineSettings);

    pub fn litert_lm_engine_settings_set_max_num_tokens(
        settings: *mut LiteRtLmEngineSettings,
        max_num_tokens: c_int,
    );

    pub fn litert_lm_engine_settings_set_cache_dir(
        settings: *mut LiteRtLmEngineSettings,
        cache_dir: *const c_char,
    );

    pub fn litert_lm_engine_create(settings: *const LiteRtLmEngineSettings) -> *mut LiteRtLmEngine;

    pub fn litert_lm_engine_delete(engine: *mut LiteRtLmEngine);

    pub fn litert_lm_conversation_config_create(
        engine: *mut LiteRtLmEngine,
        session_config: *const LiteRtLmSessionConfig,
        system_message_json: *const c_char,
        tools_json: *const c_char,
        messages_json: *const c_char,
        enable_constrained_decoding: bool,
    ) -> *mut LiteRtLmConversationConfig;

    pub fn litert_lm_conversation_config_delete(config: *mut LiteRtLmConversationConfig);

    pub fn litert_lm_conversation_create(
        engine: *mut LiteRtLmEngine,
        config: *mut LiteRtLmConversationConfig,
    ) -> *mut LiteRtLmConversation;

    pub fn litert_lm_conversation_delete(conversation: *mut LiteRtLmConversation);

    pub fn litert_lm_conversation_send_message(
        conversation: *mut LiteRtLmConversation,
        message_json: *const c_char,
        extra_context: *const c_char,
    ) -> *mut LiteRtLmJsonResponse;

    pub fn litert_lm_conversation_send_message_stream(
        conversation: *mut LiteRtLmConversation,
        message_json: *const c_char,
        extra_context: *const c_char,
        callback: LiteRtLmStreamCallback,
        callback_data: *mut c_void,
    ) -> c_int;

    pub fn litert_lm_conversation_cancel_process(conversation: *mut LiteRtLmConversation);

    pub fn litert_lm_json_response_delete(response: *mut LiteRtLmJsonResponse);

    pub fn litert_lm_json_response_get_string(
        response: *const LiteRtLmJsonResponse,
    ) -> *const c_char;
}
