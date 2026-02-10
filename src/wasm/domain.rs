//! Domain checking for WASM security
//!
//! By default, all domains are allowed. To enable domain restriction, set the
//! LINREG_DOMAIN_RESTRICT environment variable at build time:
//!
//!   LINREG_DOMAIN_RESTRICT=example.com,yoursite.com wasm-pack build
//!
//! Example for jesse-anderson.net:
//!   LINREG_DOMAIN_RESTRICT=jesse-anderson.net,tools.jesse-anderson.net,localhost,127.0.0.1 wasm-pack build
//!
//! This allows downstream users to use the library without modification while
//! still providing domain restriction as an opt-in security feature.

#![cfg(feature = "wasm")]

use wasm_bindgen::prelude::*;

use crate::error::{Error, Result};

/// Checks if the current domain is authorized to use this WASM module.
///
/// # Returns
///
/// - `Ok(())` if domain is authorized or restriction is disabled
/// - `Err(Error::DomainCheck)` if domain is not in the allowed list
pub fn check_domain() -> Result<()> {
    // Read allowed domains from build-time environment variable
    let allowed_domains = option_env!("LINREG_DOMAIN_RESTRICT");

    match allowed_domains {
        Some(domains) if !domains.is_empty() => {
            // Domain restriction is enabled
            let window =
                web_sys::window().ok_or(Error::DomainCheck("No window found".to_string()))?;
            let location = window.location();
            let hostname = location
                .hostname()
                .map_err(|_| Error::DomainCheck("No hostname found".to_string()))?;

            let domain_list: Vec<&str> = domains.split(',').map(|s| s.trim()).collect();

            if domain_list.contains(&hostname.as_str()) {
                Ok(())
            } else {
                Err(Error::DomainCheck(format!(
                    "Unauthorized domain: {}. Allowed: {}",
                    hostname, domains
                )))
            }
        },
        _ => {
            // No restriction - allow all domains
            Ok(())
        },
    }
}
