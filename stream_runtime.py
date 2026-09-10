"""Bounded NDJSON stream with cancellation and a redaction look-behind."""

import json,queue,re,threading,time

from security_guard import sanitize_output



def safe_prefix(raw):

    text,_=sanitize_output(raw)

    # Hold incomplete words and enough look-behind for credential/role markers.

    boundary=max(0,len(text)-96)

    words=list(re.finditer(r'\s',text[:boundary]))

    return text[:words[-1].start()] if words else ''



def response_stream(run,event,finish):

    messages=queue.Queue(maxsize=16)

    def publish(data):

        while not event.is_set():

            try:messages.put(data,timeout=.25);return

            except queue.Full:continue

    def worker():

        try:

            result=run(lambda text,count:publish({'type':'progress','text':safe_prefix(text),'generated_tokens':count}))

            # Completion must remain deliverable after an explicit stop.

            while True:

                try:messages.put({'type':'done','result':result},timeout=.25);break

                except queue.Full:

                    if event.is_set():break

        except Exception:publish({'type':'error','error':'Die Berechnung wurde unterbrochen. Bitte erneut versuchen.'})

        finally:finish()

    thread=threading.Thread(target=worker,daemon=True);thread.start()

    def events():

        try:

            yield json.dumps({'type':'start'})+'\n'

            while True:

                try:item=messages.get(timeout=1)

                except queue.Empty:

                    if not thread.is_alive():break

                    yield json.dumps({'type':'heartbeat'})+'\n';continue

                yield json.dumps(item,ensure_ascii=True)+'\n'

                if item['type'] in ['done','error']:break

        finally:event.set()

    return events()

