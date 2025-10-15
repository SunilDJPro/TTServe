"""
Async queue manager for handling concurrent TTS requests
"""

import asyncio
import logging
from typing import Callable, Any, Optional
from ..exceptions import QueueError

logger = logging.getLogger(__name__)


class AsyncQueueManager:
    """Manages async request queue with worker pool"""
    
    def __init__(self, max_workers: int = 1):
        self.max_workers = max_workers
        self.queue: Optional[asyncio.Queue] = None
        self.workers: list[asyncio.Task] = []
        self.is_running = False
        self._lock = asyncio.Lock()
        logger.info(f"AsyncQueueManager created with {max_workers} workers")
    
    async def start(self):
        """Start worker tasks"""
        async with self._lock:
            if self.is_running:
                logger.warning("Queue manager already running")
                return
            
            self.queue = asyncio.Queue()
            self.is_running = True
            
            for i in range(self.max_workers):
                worker = asyncio.create_task(self._worker(i))
                self.workers.append(worker)
            
            logger.info(f"Started {self.max_workers} worker(s)")
    
    async def stop(self):
        """Stop all workers"""
        async with self._lock:
            if not self.is_running:
                return
            
            self.is_running = False
            
            # Cancel all workers
            for worker in self.workers:
                worker.cancel()
            
            # Wait for cancellation
            await asyncio.gather(*self.workers, return_exceptions=True)
            
            self.workers.clear()
            self.queue = None
            
            logger.info("All workers stopped")
    
    async def _worker(self, worker_id: int):
        """Worker task that processes queue items"""
        logger.info(f"Worker {worker_id} started")
        
        while self.is_running:
            try:
                task_func, future = await self.queue.get()
                logger.debug(f"Worker {worker_id} processing task")
                
                try:
                    result = await task_func()
                    future.set_result(result)
                except Exception as e:
                    logger.error(f"Worker {worker_id} task failed: {e}")
                    future.set_exception(e)
                finally:
                    self.queue.task_done()
                    
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(0.1)
    
    async def submit(self, task_func: Callable) -> Any:
        """Submit a task to the queue and wait for result"""
        if not self.is_running or self.queue is None:
            raise QueueError("Queue manager not running")
        
        future = asyncio.Future()
        await self.queue.put((task_func, future))
        
        logger.debug(f"Task submitted to queue (size: {self.queue.qsize()})")
        
        return await future
    
    @property
    def queue_size(self) -> int:
        """Get current queue size"""
        if self.queue is None:
            return 0
        return self.queue.qsize()
    
    @property
    def active_workers(self) -> int:
        """Get number of active workers"""
        return len([w for w in self.workers if not w.done()])