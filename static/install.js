'use strict';
let installPrompt=null;
window.addEventListener('beforeinstallprompt',e=>{e.preventDefault();installPrompt=e;});
document.getElementById('installApp').onclick=async()=>{
 if(installPrompt){await installPrompt.prompt();await installPrompt.userChoice;installPrompt=null;}
 else notify(matchMedia('(display-mode: standalone)').matches?'Neral ist bereits als App geöffnet.':'Im Browser-Menü „App installieren“ wählen. Android: die gesicherte Handy-Adresse vom PC verwenden.');
};
if('serviceWorker' in navigator&&window.isSecureContext)navigator.serviceWorker.register('/sw.js').catch(()=>{});
window.addEventListener('offline',()=>{ready=false;modelReady=false;setBusy(busy);notify('Offline. Deine Chats bleiben auf diesem Gerät. Neue Antworten benötigen die Verbindung zum PC.');});
window.addEventListener('online',()=>reconnect());
