"""
Unit tests for TLS server functionality
Tests encryption, certificate handling, and secure communication
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import ssl
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from servers.base_tls_server import TLSConfig, BaseTLSDynamicServer


class TestTLSConfig(unittest.TestCase):
    """Test TLS configuration"""
    
    def test_default_config(self):
        """Test default TLS configuration"""
        config = TLSConfig()
        self.assertEqual(config.certfile, 'certs/server.crt')
        self.assertEqual(config.keyfile, 'certs/server.key')
        self.assertIsNone(config.cafile)
        self.assertFalse(config.require_client_cert)
    
    def test_custom_config(self):
        """Test custom TLS configuration"""
        config = TLSConfig(
            certfile='custom.crt',
            keyfile='custom.key',
            cafile='ca.crt',
            require_client_cert=True
        )
        self.assertEqual(config.certfile, 'custom.crt')
        self.assertEqual(config.keyfile, 'custom.key')
        self.assertEqual(config.cafile, 'ca.crt')
        self.assertTrue(config.require_client_cert)
    
    @patch.dict(os.environ, {
        'TLS_CERT_FILE': 'env_cert.crt',
        'TLS_KEY_FILE': 'env_key.key',
        'TLS_CA_FILE': 'env_ca.crt'
    })
    def test_env_config(self):
        """Test TLS configuration from environment variables"""
        config = TLSConfig()
        self.assertEqual(config.certfile, 'env_cert.crt')
        self.assertEqual(config.keyfile, 'env_key.key')
        self.assertEqual(config.cafile, 'env_ca.crt')


class TestBaseTLSServer(unittest.TestCase):
    """Test base TLS server functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock certificate files to avoid file system operations
        with patch('pathlib.Path.exists', return_value=True):
            self.server = BaseTLSDynamicServer(
                host='localhost',
                port=8911,
                server_name='Test Server'
            )
    
    def test_ssl_context_creation(self):
        """Test SSL context creation"""
        with patch('ssl.create_default_context') as mock_ssl:
            mock_context = MagicMock()
            mock_ssl.return_value = mock_context
            
            context = self.server.create_ssl_context()
            
            # Verify SSL context configuration
            mock_ssl.assert_called_once_with(ssl.Purpose.CLIENT_AUTH)
            mock_context.load_cert_chain.assert_called_once()
            mock_context.set_ciphers.assert_called_once()
            
            # Check minimum TLS version
            self.assertEqual(mock_context.minimum_version, ssl.TLSVersion.TLSv1_2)
    
    def test_mutual_tls_configuration(self):
        """Test mutual TLS configuration"""
        with patch('pathlib.Path.exists', return_value=True):
            server = BaseTLSDynamicServer(
                host='localhost',
                port=8911,
                server_name='Test Server',
                tls_config=TLSConfig(require_client_cert=True, cafile='ca.crt')
            )
        
        with patch('ssl.create_default_context') as mock_ssl:
            mock_context = MagicMock()
            mock_ssl.return_value = mock_context
            
            context = server.create_ssl_context()
            
            # Verify mutual TLS settings
            self.assertEqual(mock_context.verify_mode, ssl.CERT_REQUIRED)
            mock_context.load_verify_locations.assert_called_once_with(cafile='ca.crt')
    
    def test_cipher_suites(self):
        """Test that only strong cipher suites are used"""
        with patch('ssl.create_default_context') as mock_ssl:
            mock_context = MagicMock()
            mock_ssl.return_value = mock_context
            
            self.server.create_ssl_context()
            
            # Verify strong ciphers are set
            expected_ciphers = 'ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS'
            mock_context.set_ciphers.assert_called_once_with(expected_ciphers)
    
    @patch('socket.socket')
    @patch('ssl.SSLSocket')
    def test_client_handler_encryption(self, mock_ssl_socket, mock_socket):
        """Test that client connections are encrypted"""
        # Mock encrypted socket
        mock_ssl_socket.recv.return_value = b"1,test query\n"
        mock_ssl_socket.send.return_value = None
        
        # Mock process_batch to return a result
        self.server.process_batch = Mock(return_value=['test_result'])
        
        # Test client handler
        self.server.handle_client(mock_ssl_socket, ('127.0.0.1', 12345))
        
        # Verify encrypted communication
        mock_ssl_socket.recv.assert_called()
        mock_ssl_socket.send.assert_called()
    
    def test_ssl_error_handling(self):
        """Test handling of SSL errors"""
        mock_socket = MagicMock()
        mock_socket.recv.side_effect = ssl.SSLError("SSL handshake failed")
        
        # Should handle SSL error gracefully
        self.server.handle_client(mock_socket, ('127.0.0.1', 12345))
        
        # Verify socket is closed
        mock_socket.close.assert_called()


class TestCertificateValidation(unittest.TestCase):
    """Test certificate validation and security"""
    
    @patch('subprocess.run')
    @patch('pathlib.Path.exists', return_value=False)
    def test_self_signed_cert_generation(self, mock_exists, mock_subprocess):
        """Test self-signed certificate generation"""
        config = TLSConfig()
        
        # Verify certificate generation commands
        calls = mock_subprocess.call_args_list
        
        # Check private key generation
        self.assertIn('genrsa', str(calls[0]))
        self.assertIn('2048', str(calls[0]))
        
        # Check certificate generation
        self.assertIn('req', str(calls[1]))
        self.assertIn('-x509', str(calls[1]))
        self.assertIn('-days', str(calls[1]))
    
    def test_certificate_permissions(self):
        """Test that private keys have secure permissions"""
        with patch('subprocess.run'):
            with patch('pathlib.Path.exists', return_value=False):
                with patch('os.chmod') as mock_chmod:
                    config = TLSConfig()
                    
                    # Verify secure permissions (0o600 = read/write for owner only)
                    mock_chmod.assert_called_with(config.keyfile, 0o600)


class TestSecureCommunication(unittest.TestCase):
    """Test secure communication patterns"""
    
    def test_no_plaintext_fallback(self):
        """Ensure server never falls back to plaintext"""
        with patch('pathlib.Path.exists', return_value=True):
            server = BaseTLSDynamicServer(
                host='localhost',
                port=8911,
                server_name='Test Server'
            )
        
        # Mock a failed SSL handshake
        mock_socket = MagicMock()
        mock_socket.recv.side_effect = ssl.SSLError("Handshake failed")
        
        # Server should not attempt plaintext communication
        server.handle_client(mock_socket, ('127.0.0.1', 12345))
        
        # Verify no plaintext data is sent
        for call in mock_socket.send.call_args_list:
            self.assertIsInstance(call, (type(None), MagicMock))
    
    def test_tls_version_enforcement(self):
        """Test that old TLS versions are rejected"""
        with patch('pathlib.Path.exists', return_value=True):
            server = BaseTLSDynamicServer(
                host='localhost',
                port=8911,
                server_name='Test Server'
            )
        
        with patch('ssl.create_default_context') as mock_ssl:
            mock_context = MagicMock()
            mock_ssl.return_value = mock_context
            
            server.create_ssl_context()
            
            # Verify TLS 1.2 or higher is enforced
            self.assertEqual(mock_context.minimum_version, ssl.TLSVersion.TLSv1_2)


if __name__ == '__main__':
    unittest.main()