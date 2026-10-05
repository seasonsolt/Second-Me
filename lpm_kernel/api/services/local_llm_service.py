import os
import json
import logging
import psutil
import time
import subprocess
import platform
import re
import socket
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx
import torch  # Add torch import for CUDA detection
import threading
import queue
from typing import Iterator, Any, Optional, Generator, Dict
from datetime import datetime
from flask import Response
from openai import OpenAI
from lpm_kernel.api.domains.kernel2.dto.server_dto import ServerStatus, ProcessInfo
from lpm_kernel.configs.config import Config
import uuid

logger = logging.getLogger(__name__)

class LocalLLMService:
    """Service for managing local LLM client and server"""
    
    def __init__(self):
        self._client = None
        self._stopping_server = False
        self._process = None
        self._server_log = None
        
    @property
    def client(self) -> OpenAI:
        config = Config.from_env()
        """Get the OpenAI client for local LLM server"""
        if self._client is None:
            base_url = config.get("LOCAL_LLM_SERVICE_URL")
            if not base_url:
                raise ValueError("LOCAL_LLM_SERVICE_URL environment variable is not set")
                
            self._client = OpenAI(
                base_url=base_url,
                api_key="sk-no-key-required"
            )
        return self._client

    def _server_settings(self):
        config = Config.from_env()
        url = urlsplit(config.get("LOCAL_LLM_SERVICE_URL", "http://127.0.0.1:8080/v1"))
        port = url.port or (443 if url.scheme == "https" else 8080)
        origin = urlunsplit((url.scheme, url.netloc, "", "", ""))
        context_size = int(config.get("LOCAL_LLM_CONTEXT_SIZE", "8192"))
        parallel = int(config.get("LOCAL_LLM_PARALLEL", "1"))
        if context_size < 512 or parallel < 1 or context_size // parallel < 512:
            raise ValueError("Each llama-server slot needs at least 512 context tokens")
        return port, origin, context_size, parallel

    def _server_path(self):
        executable = "llama-server.exe" if os.name == "nt" else "llama-server"
        return Path.cwd() / "llama.cpp" / "build" / "bin" / executable

    def _managed_processes(self):
        server_path = self._server_path().resolve()
        port, _, _, _ = self._server_settings()
        for process in psutil.process_iter(["pid", "cmdline"]):
            try:
                args = process.cmdline()
                if not args:
                    continue
                executable = Path(args[0])
                if not executable.is_absolute():
                    executable = Path(process.cwd()) / executable
                if executable.resolve() != server_path:
                    continue
                for flag in ("--port", "-p"):
                    if flag in args and args[args.index(flag) + 1] == str(port):
                        yield process
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, IndexError):
                continue

    def _ensure_port_available(self, host, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, port))
            except OSError as exc:
                raise RuntimeError(f"Port {port} is already in use; choose another LOCAL_LLM_SERVICE_URL") from exc

    def start_server(self, model_path: str, use_gpu: bool = True) -> bool:
        """Start the pinned server for this worktree and wait for model readiness."""
        started_process = None
        started_log = None
        try:
            model = Path(model_path).resolve()
            if not model.is_file():
                raise FileNotFoundError(f"GGUF model not found: {model}")
            server_path = self._server_path()
            revision = (Path.cwd() / "dependencies" / "llama.cpp.version").read_text().strip()
            marker = server_path.parent.parent / ".secondme-build"
            build = marker.read_text().strip().split() if marker.exists() else []
            if not server_path.is_file() or len(build) != 2 or build[0] != revision:
                raise RuntimeError("llama.cpp build is missing or obsolete; rerun setup for this platform")

            status = self.get_server_status()
            if status.is_running:
                args = status.process_info.cmdline
                loaded_model = args[args.index("-m") + 1] if "-m" in args else None
                if loaded_model and Path(loaded_model).resolve() != model:
                    raise RuntimeError("Stop this worktree's current model before loading another")
                return True

            metal = use_gpu and build[1] == "metal" and platform.system() == "Darwin"
            cuda = use_gpu and build[1] == "cuda" and torch.cuda.is_available()
            backend = "Metal" if metal else "CUDA" if cuda else "CPU"
            port, origin, context_size, parallel = self._server_settings()
            # Loopback by default; Docker sets 0.0.0.0 so its port mapping reaches the server.
            host = Config.from_env().get("LLAMA_SERVER_HOST", "127.0.0.1")
            self._ensure_port_available(host, port)
            command = [
                str(server_path), "-m", str(model), "--host", host,
                "--port", str(port), "--ctx-size", str(context_size),
                "--parallel", str(parallel), "--cont-batching", "--jinja",
                "--chat-template-kwargs", '{"enable_thinking":false}',
                "--n-gpu-layers", "999" if metal or cuda else "0",
                "--threads", str(max(1, (os.cpu_count() or 2) - 1)),
            ]
            logs = Path.cwd() / "logs"
            logs.mkdir(exist_ok=True)
            log_path = logs / f"llama-server-{port}.log"
            self._server_log = log_path.open("w", encoding="utf-8")
            started_log = self._server_log
            self._process = subprocess.Popen(
                command, stdout=self._server_log, stderr=subprocess.STDOUT,
                env=os.environ.copy(), start_new_session=os.name != "nt",
            )
            started_process = self._process
            logger.info("Starting %s llama-server on port %s; log: %s", backend, port, log_path)
            deadline = time.monotonic() + 120
            with httpx.Client(timeout=1, trust_env=False) as client:
                while time.monotonic() < deadline:
                    if self._process.poll() is not None:
                        raise RuntimeError(f"llama-server exited; see {log_path}")
                    try:
                        if (client.get(origin + "/health").status_code == 200
                                and self._process.poll() is None):
                            logger.info("llama-server ready (%s), context=%s, parallel=%s", backend, context_size, parallel)
                            return True
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.2)
            raise TimeoutError(f"llama-server did not become ready; see {log_path}")
        except Exception as exc:
            logger.error("Error starting llama-server: %s", exc)
            if started_process is not None and started_process.poll() is None:
                started_process.terminate()
                try:
                    started_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    started_process.kill()
                    started_process.wait()
            if started_log is not None:
                started_log.close()
            if started_log is self._server_log:
                self._server_log = None
            if started_process is self._process:
                self._process = None
            return False

    def stop_server(self) -> ServerStatus:
        """Stop only the server executable and port belonging to this worktree."""
        if self._stopping_server:
            return self.get_server_status()
        self._stopping_server = True
        try:
            for process in self._managed_processes():
                try:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            if self._process is not None:
                self._process.poll()
            if self._server_log is not None:
                self._server_log.close()
                self._server_log = None
            self._process = None
            return self.get_server_status()
        finally:
            self._stopping_server = False

    def get_server_status(self) -> ServerStatus:
        try:
            for process in self._managed_processes():
                with process.oneshot():
                    return ServerStatus.running(ProcessInfo(
                        pid=process.pid,
                        cpu_percent=process.cpu_percent(),
                        memory_percent=process.memory_percent(),
                        create_time=process.create_time(),
                        cmdline=process.cmdline(),
                    ))
        except (psutil.Error, OSError, ValueError) as exc:
            logger.error("Error checking llama-server status: %s", exc)
        return ServerStatus.not_running()

    def _count_prompt_tokens(self, client, origin, messages):
        response = client.post(origin + "/apply-template", json={
            "messages": messages,
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        response.raise_for_status()
        prompt = response.json()["prompt"]
        response = client.post(origin + "/tokenize", json={
            "content": prompt, "add_special": True, "parse_special": True,
        })
        response.raise_for_status()
        return len(response.json()["tokens"])

    def prepare_chat_request(self, messages, max_tokens):
        """Fit the real model template, history and references in one server slot."""
        _, origin, context_size, parallel = self._server_settings()
        capacity = context_size // parallel
        output_tokens = min(max(1, max_tokens), capacity // 4)
        budget = capacity - output_tokens - 32
        fitted = [dict(message) for message in messages]
        last_user = next((i for i in range(len(fitted) - 1, -1, -1)
                          if fitted[i].get("role") == "user"), None)
        if last_user is None:
            raise ValueError("A user message is required")
        with httpx.Client(timeout=10, trust_env=False) as client:
            while self._count_prompt_tokens(client, origin, fitted) > budget:
                last_user = max(i for i, message in enumerate(fitted) if message.get("role") == "user")
                history = next((i for i in range(last_user) if fitted[i].get("role") != "system"), None)
                if history is not None:
                    del fitted[history]
                    # Remove the corresponding assistant turn, not the current question.
                    while history < len(fitted) and fitted[history].get("role") == "assistant":
                        del fitted[history]
                    continue
                reduced = False
                for message in fitted:
                    if message.get("role") != "system":
                        continue
                    match = re.search(r"<reference_memories>(.*?)</reference_memories>", message["content"], re.DOTALL)
                    if match is None:
                        continue
                    references = re.findall(r'(<memory source="[^"]*">)(.*?)(</memory>)', match.group(1), re.DOTALL)
                    if not references:
                        continue
                    # Remove the least-prioritized reference first; shorten the last one if needed.
                    if len(references) > 1:
                        references.pop()
                        region = "\n".join("".join(reference) for reference in references)
                    else:
                        opening, content, closing = references[0]
                        if len(content) > 128:
                            region = opening + content[:len(content) // 2].rsplit("&", 1)[0] + "…" + closing
                        else:
                            region = "Relevant memories omitted to fit the context limit."
                    message["content"] = message["content"][:match.start(1)] + "\n" + region + "\n" + message["content"][match.end(1):]
                    reduced = True
                    break
                if not reduced:
                    raise ValueError("Current question and system rules exceed the model context limit; shorten the question")
        return fitted, output_tokens

    def _parse_response_chunk(self, chunk):
        """Parse different response chunk formats into a standardized format."""
        try:
            if chunk is None:
                logger.warning("Received None chunk")
                return None
                
            # logger.info(f"Parsing response chunk: {chunk}")
            # Handle custom format
            if isinstance(chunk, dict) and "type" in chunk and chunk["type"] == "chat_response":
                logger.info(f"Processing custom format response: {chunk}")
                return {
                    "id": str(uuid.uuid4()),  # Generate a unique ID
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now().timestamp()),
                    "model": "models/lpm",
                    "system_fingerprint": None,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": chunk.get("content", "")
                            },
                            "finish_reason": "stop" if chunk.get("done", False) else None
                        }
                    ]
                }
            
            # Handle OpenAI format
            if not hasattr(chunk, 'choices'):
                logger.warning(f"Chunk has no choices attribute: {chunk}")
                return None
                
            choices = getattr(chunk, 'choices', [])
            if not choices:
                logger.warning("Chunk has empty choices")
                return None
                
            # logger.info(f"Processing OpenAI format response: choices={choices}")
            delta = choices[0].delta
            
            # Create standard response structure
            response_data = {
                "id": chunk.id,
                "object": "chat.completion.chunk",
                "created": int(datetime.now().timestamp()),
                "model": "models/lpm",
                "system_fingerprint": chunk.system_fingerprint if hasattr(chunk, 'system_fingerprint') else None,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            # Keep even if content is None, let the client handle it
                            "content": delta.content if hasattr(delta, 'content') else ""
                        },
                        "finish_reason": choices[0].finish_reason
                    }
                ]
            }
            
            # If there is neither content nor finish_reason, skip
            if not (hasattr(delta, 'content') or choices[0].finish_reason):
                logger.debug("Skipping chunk with no content and no finish_reason")
                return None
                
            return response_data
            
        except Exception as e:
            logger.error(f"Error parsing response chunk: {e}, chunk: {chunk}")
            return None

    def handle_stream_response(self, response_iter: Iterator[Any]) -> Response:
        """Handle streaming response from the LLM server"""
        # Create a queue for thread communication
        message_queue = queue.Queue()
        # Create an event flag to notify when model processing is complete
        completion_event = threading.Event()
        # Create a variable to track if heartbeat is needed after first response
        first_response_received = False
        
        def heartbeat_thread():
            """Thread function for sending heartbeats"""
            start_time = time.time()
            heartbeat_interval = 10  # Send heartbeat every 10 seconds
            heartbeat_count = 0
            
            logger.info("[STREAM_DEBUG] Heartbeat thread started")
            
            try:
                # Send initial heartbeat
                message_queue.put((b": initial heartbeat\n\n", "[INITIAL_HEARTBEAT]"))
                last_heartbeat_time = time.time()
                
                while not completion_event.is_set():
                    current_time = time.time()
                    
                    # Check if we need to send a heartbeat
                    if current_time - last_heartbeat_time >= heartbeat_interval:
                        heartbeat_count += 1
                        elapsed = current_time - start_time
                        logger.info(f"[STREAM_DEBUG] Sending heartbeat #{heartbeat_count} at {elapsed:.2f}s")
                        message_queue.put((f": heartbeat #{heartbeat_count}\n\n".encode('utf-8'), "[HEARTBEAT]"))
                        last_heartbeat_time = current_time
                    
                    # Short sleep to prevent CPU spinning
                    time.sleep(0.1)
                
                logger.info(f"[STREAM_DEBUG] Heartbeat thread stopping after {heartbeat_count} heartbeats")
            except Exception as e:
                logger.error(f"[STREAM_DEBUG] Error in heartbeat thread: {str(e)}", exc_info=True)
                message_queue.put((f"data: {{\"error\": \"Heartbeat error: {str(e)}\"}}\n\n".encode('utf-8'), "[ERROR]"))
        
        def model_response_thread():
            """Thread function for processing model responses"""
            chunk = None
            start_time = time.time()
            chunk_count = 0
            
            try:
                logger.info("[STREAM_DEBUG] Model response thread started")
                
                # Process model responses
                for chunk in response_iter:
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    chunk_count += 1
                    
                    logger.info(f"[STREAM_DEBUG] Received chunk #{chunk_count} after {elapsed_time:.2f}s")
                    
                    if chunk is None:
                        logger.warning("[STREAM_DEBUG] Received None chunk, skipping")
                        continue
                    
                    # Check if it's an end marker
                    if chunk == "[DONE]":
                        logger.info(f"[STREAM_DEBUG] Received [DONE] marker after {elapsed_time:.2f}s")
                        message_queue.put((b"data: [DONE]\n\n", "[DONE]"))
                        break
                    
                    # Handle error responses
                    if isinstance(chunk, dict) and "error" in chunk:
                        logger.warning(f"[STREAM_DEBUG] Received error response: {chunk}")
                        data_str = json.dumps(chunk)
                        message_queue.put((f"data: {data_str}\n\n".encode('utf-8'), "[ERROR]"))
                        message_queue.put((b"data: [DONE]\n\n", "[DONE]"))
                        break
                    
                    # Handle normal responses
                    response_data = self._parse_response_chunk(chunk)
                    if response_data:
                        data_str = json.dumps(response_data)
                        content = response_data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        content_length = len(content) if content else 0
                        logger.info(f"[STREAM_DEBUG] Sending chunk #{chunk_count}, content length: {content_length}, elapsed: {elapsed_time:.2f}s")
                        message_queue.put((f"data: {data_str}\n\n".encode('utf-8'), "[CONTENT]"))
                    else:
                        logger.warning(f"[STREAM_DEBUG] Parsed response data is None for chunk #{chunk_count}")
                
                # Handle the case where no responses were received
                if chunk_count == 0:
                    logger.info("[STREAM_DEBUG] No chunks received, sending empty message")
                    thinking_message = {
                        "id": str(uuid.uuid4()),
                        "object": "chat.completion.chunk",
                        "created": int(datetime.now().timestamp()),
                        "model": "models/lpm",
                        "system_fingerprint": None,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": ""  # Empty content won't affect frontend display
                                },
                                "finish_reason": None
                            }
                        ]
                    }
                    data_str = json.dumps(thinking_message)
                    message_queue.put((f"data: {data_str}\n\n".encode('utf-8'), "[THINKING]"))
                
                # Model processing is complete, send end marker
                if chunk != "[DONE]":
                    logger.info(f"[STREAM_DEBUG] Sending final [DONE] marker after {elapsed_time:.2f}s")
                    message_queue.put((b"data: [DONE]\n\n", "[DONE]"))
                
            except Exception as e:
                logger.error(f"[STREAM_DEBUG] Error processing model response: {str(e)}", exc_info=True)
                message_queue.put((f"data: {{\"error\": \"{str(e)}\"}}\n\n".encode('utf-8'), "[ERROR]"))
                message_queue.put((b"data: [DONE]\n\n", "[DONE]"))
            finally:
                # Set completion event to notify heartbeat thread to stop
                completion_event.set()
                logger.info(f"[STREAM_DEBUG] Model response thread completed with {chunk_count} chunks")
        
        def generate():
            """Main generator function for generating responses"""
            # Start heartbeat thread
            heart_thread = threading.Thread(target=heartbeat_thread, daemon=True)
            heart_thread.start()
            
            # Start model response processing thread
            model_thread = threading.Thread(target=model_response_thread, daemon=True)
            model_thread.start()
            
            try:
                # Get messages from queue and return to client
                while True:
                    try:
                        # Use short timeout to get message, prevent blocking
                        message, message_type = message_queue.get(timeout=0.1)
                        logger.debug(f"[STREAM_DEBUG] Yielding message type: {message_type}")
                        yield message
                        
                        # If end marker is received, exit loop
                        if message_type == "[DONE]":
                            logger.info("[STREAM_DEBUG] Received [DONE] marker, ending generator")
                            break
                    except queue.Empty:
                        # Queue is empty, continue trying to get message
                        # Check if model thread has completed but didn't send [DONE]
                        if completion_event.is_set() and not model_thread.is_alive():
                            logger.warning("[STREAM_DEBUG] Model thread completed without [DONE], ending generator")
                            yield b"data: [DONE]\n\n"
                            break
                        pass
            except GeneratorExit:
                # Client closed connection
                logger.info("[STREAM_DEBUG] Client closed connection (GeneratorExit)")
                completion_event.set()
            except Exception as e:
                logger.error(f"[STREAM_DEBUG] Error in generator: {str(e)}", exc_info=True)
                try:
                    yield f"data: {{\"error\": \"Generator error: {str(e)}\"}}\n\n".encode('utf-8')
                    yield b"data: [DONE]\n\n"
                except:
                    pass
                completion_event.set()
            finally:
                # Ensure completion event is set
                completion_event.set()
                # Wait for threads to complete
                if heart_thread.is_alive():
                    heart_thread.join(timeout=1.0)
                if model_thread.is_alive():
                    model_thread.join(timeout=1.0)
                logger.info("[STREAM_DEBUG] Generator completed")
        
        # Return response
        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache, no-transform',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive',
                'Transfer-Encoding': 'chunked'
            }
        )


# Global instance
local_llm_service = LocalLLMService()
