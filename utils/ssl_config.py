"""
SSL Configuration Utility - Configure SSL certificate handling once for all modules

This module MUST be imported FIRST before any HTTP libraries are imported.
It configures SSL verification based on environment variables and runs only once.
"""

import os
import ssl
from pathlib import Path

# Track if SSL has been configured to ensure it only runs once
_ssl_configured = False


def configure_ssl():
    """
    Configure SSL certificate handling based on environment variables.
    This function is idempotent - it only runs once even if called multiple times.
    
    Environment variables:
    - DISABLE_SSL_VERIFY: Set to "1", "true", or "yes" to disable SSL verification
    - HF_HUB_DISABLE_SSL_VERIFICATION: Set to "1", "true", or "yes" to disable SSL for HuggingFace
    - SSL_CERT_FILE: Path to custom SSL certificate file
    """
    global _ssl_configured
    
    # Only configure once
    if _ssl_configured:
        return
    
    _ssl_configured = True
    
    # Read environment variables
    cert_file = os.getenv("SSL_CERT_FILE", "")
    disable_ssl = os.getenv("DISABLE_SSL_VERIFY", "").lower() in ("1", "true", "yes")
    hf_disable_ssl = os.getenv("HF_HUB_DISABLE_SSL_VERIFICATION", "").lower() in (
        "1",
        "true",
        "yes",
    )
    
    # If HuggingFace SSL is disabled, treat it as disable SSL
    if hf_disable_ssl:
        disable_ssl = True
    
    # Disable SSL immediately if needed, before any other imports
    if disable_ssl or hf_disable_ssl:
        _configure_ssl_disable()
        if cert_file:
            print(f"⚠ Note: SSL_CERT_FILE is set but SSL verification is disabled")
    elif cert_file:
        # Use custom certificate file
        cert_path = Path(cert_file)
        if cert_path.exists():
            cert_abs_path = str(cert_path.resolve())
            # Set for requests library
            os.environ["REQUESTS_CA_BUNDLE"] = cert_abs_path
            os.environ["CURL_CA_BUNDLE"] = cert_abs_path
            # Set for Python's ssl module
            os.environ["SSL_CERT_FILE"] = cert_abs_path
            # Also configure ssl context with the certificate
            try:
                context = ssl.create_default_context(cafile=cert_abs_path)
                ssl._create_default_https_context = lambda: context
            except Exception as e:
                print(f"⚠ Warning: Could not load certificate for ssl module: {e}")
                print("⚠ Falling back to disabling SSL verification")
                _configure_ssl_disable()
            print(f"Using custom SSL certificate: {cert_abs_path}")
        else:
            print(f"⚠ Warning: Certificate file not found: {cert_path}")
            print("⚠ Falling back to disabling SSL verification")
            _configure_ssl_disable()


def _configure_ssl_disable():
    """Configure SSL to be disabled for all HTTP libraries"""
    # Disable SSL verification for Python's ssl module
    ssl._create_default_https_context = ssl._create_unverified_context
    # Set environment variables
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["SSL_CERT_FILE"] = ""
    os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
    
    # Disable urllib3 warnings and verification
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        # Patch urllib3 to not verify SSL
        try:
            urllib3.util.ssl_.create_urllib3_context = (
                lambda *args, **kwargs: ssl._create_unverified_context()
            )
        except AttributeError:
            pass
    except ImportError:
        pass
    
    # Patch requests if available
    try:
        import requests
        # Disable SSL verification for requests
        requests.packages.urllib3.disable_warnings()
        # Monkey patch requests.Session to not verify SSL
        original_request = requests.Session.request
        
        def patched_request(self, *args, **kwargs):
            kwargs.setdefault("verify", False)
            return original_request(self, *args, **kwargs)
        
        requests.Session.request = patched_request
    except (ImportError, AttributeError):
        pass
    
    # Patch httpx if available (HuggingFace might use httpx)
    try:
        import httpx
        # Create a custom transport that doesn't verify SSL
        original_transport = httpx.HTTPTransport
        
        def create_unverified_transport(*args, **kwargs):
            kwargs["verify"] = False
            return original_transport(*args, **kwargs)
        
        httpx.HTTPTransport = create_unverified_transport
    except (ImportError, AttributeError):
        pass


# Auto-configure SSL when module is imported
configure_ssl()
























