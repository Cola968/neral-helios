"use strict";
// Browser capture stays on this machine. The script only forwards raw PCM
// samples to its same-origin UI; it never sends network requests by itself.
class NeralLocalPcmCapture extends AudioWorkletProcessor {
  constructor(){
    super();
    this.buffer=new Float32Array(2048);
    this.index=0;
  }
  process(inputs,outputs){
    const mono=inputs[0]?.[0];
    if(mono){
      for(let i=0;i<mono.length;i++){
        this.buffer[this.index++]=mono[i];
        if(this.index===this.buffer.length){
          const pcm=this.buffer;
          this.buffer=new Float32Array(2048);
          this.index=0;
          this.port.postMessage(pcm,[pcm.buffer]);
        }
      }
    }
    // Do not play microphone audio through the speakers (echo loop).
    for(const channel of outputs[0]||[])channel.fill(0);
    return true;
  }
}
registerProcessor("neral-local-capture",NeralLocalPcmCapture);
