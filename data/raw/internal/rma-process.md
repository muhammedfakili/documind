# Warranty Replacement (RMA) Process (INTERNAL)

**Owner:** Customer Operations
**Last updated:** May 12, 2026

An RMA (Return Merchandise Authorization) is created when a product is defective and still under warranty. Check the warranty period in the warranty matrix before starting.

## Steps

1. **Verify the serial number.** Ask the customer for the serial number (printed inside the charging case lid for earbuds, under the ear cushion for headphones, on the base of speakers). Look it up in the Device Registry to confirm the purchase date and region.
2. **Run basic troubleshooting first**, using the product's user manual. Record which steps the customer already tried.
3. **Create the RMA** with the `RMA-start` macro. Choose the reason code:
   - `DEF-AUDIO` no sound or distorted sound
   - `DEF-BATT` battery or charging defect
   - `DEF-CONN` Bluetooth or pairing defect
   - `DEF-WATER` water ingress on a product with an IP rating (see warranty terms for conditions)
   - `DEF-OTHER` anything else, with a written description
4. **Choose the fulfilment type:**
   - *Advance replacement* (we ship the new unit first) for customers in the US, UK and EU when the product was bought directly from Lumora.
   - *Standard replacement* (customer ships first, we send the replacement after inspection) in all other cases.
5. The customer receives a prepaid label by email. Standard replacements ship within **3 business days** of the defective unit arriving at the warehouse.

## Earbuds: single-side replacement

For Pulse and Pulse Pro, a single defective earbud or the charging case alone can be replaced. Use the part codes `L`, `R` or `CASE` in the RMA. The replacement part inherits the remaining warranty of the original product.

## When not to create an RMA

- Physical damage from drops, crushing or pets: offer the out-of-warranty repair price list instead.
- Lost items: warranty does not cover loss. Single earbuds can be bought as spare parts.
- Suspected battery damage: follow the safety escalation in the Support Handbook instead of the normal RMA flow.
