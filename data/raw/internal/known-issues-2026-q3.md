# Known Issues — Q3 2026 (INTERNAL)

**Last updated:** September 9, 2026
Agents: check this list before creating an RMA. Many reports below are fixed by a firmware update, not by a replacement.

## Pulse (earbuds)

**E-214 reported after charging in hot environments.** Firmware 3.2.0 and earlier trigger E-214 too early when the right earbud's battery sensor reads above 43 °C, for example after charging in a car. Firmware **3.2.1** (released August 27, 2026) raises the threshold to the correct value of 45 °C. Ask the customer to update before creating an RMA. If E-214 still appears on 3.2.1 at normal room temperature, it is a genuine sensor fault: create an RMA with reason `DEF-BATT` and part code `R`.

**Left earbud drops connection in single-earbud mode on some Android 15 phones.** Workaround: disable "LE Audio" in the phone's Bluetooth developer settings. Fix planned for firmware 3.3.

## Pulse Pro (earbuds)

**Transparency mode sounds muffled after firmware 5.0.4.** Caused by wax build-up on the outer microphone mesh in most cases. Ask the customer to clean the mesh with a dry soft brush. Not a firmware bug.

## Halo ANC (headphones)

**Right ear cushion detaches.** Affects units with serial numbers starting with `HA24`. Send a free replacement cushion kit (part `HA-CUSH-2`) without an RMA.

## Aria X2 (speaker)

**Party Mode pairing fails between an Aria X2 and an Aria Mini.** Party Mode only works between two speakers of the same model. This is expected behavior, not a bug; the product page was updated on September 1, 2026.
