# Claims & Evidence

## Claim 1 — Nine of the GM's bottom 10 are Back Forty, and they are not down
- **Query:** on-floor `prepared_daily_view`, rank by sum of house win, join zone and availability.
- **Result:** 9/10 are `Z-B40`. Mean availability 99.7%. IDs: ['M-5076', 'M-1158', 'M-1120', 'M-1118', 'M-1238', 'M-1181', 'M-1035', 'M-1179', 'M-1017', 'M-1265'].
- **Alternative tested:** the machines are broken or mis-paying. Availability near 100% and theo/hr in line with the zone reject that.
- **Limitation:** zone labels come from the last placement in the window.

## Claim 2 — M-1158 is a luck event, not a bad box
- **Query:** house win, theo, luck residual, theo per available hour vs zone median.
- **Result:** luck residual $-36,165. Theo/hr $11.97 vs zone $11.96.
- **Alternative tested:** the paytable is wrong. Theo uses configured RTP; the gap is residual, not theo.
- **Limitation:** six months is still a short sample on a medium-volatility game.

## Claim 3 — Back Forty earns about $6/hr less theo than the fleet
- **Query:** median theo per available hour by zone.
- **Result:** Back Forty $11.96/hr vs fleet $18.15/hr vs Main Entrance $21.69/hr.
- **Alternative tested:** the zone is full of low-RTP games only. The same titles earn more theo/hr in Center Pit and Main Entrance.
- **Limitation:** we do not observe whether guests avoid the zone or the zone is physically hard to find.

## Claim 4 — Downtime is real and concentrated, and still too small for a July tech hire
- **Query:** fault_repair duration × machine theo/hr; faults per machine by manufacturer.
- **Result:** $88,347 lost theo on 1207 faults (~$488/day). Ridgeline 7.2 faults/machine vs Halcyon 1.9.
- **Alternative tested:** buy a technician. Lead time leaves ~23 days; recovery does not cover $11,520.
- **Limitation:** lost theo assumes the machine would have earned its usual theo/hr while down.

## Claim 5 — A weekday mean beats a flat mean, and that kills the promo
- **Query:** `property_day` coin-in. Train through 2025-04-30. Hold out May–June.
- **Result:** weekday MAE $68,699 vs baseline $218,258. Weekend premium $358,881/day.
- **Decision:** Weekday model beats the constant mean (MAE $68,699 vs $218,258). Adding announced events makes holdout MAE worse ($68,738), so events do not earn a promo. Weekends already run ~$358,881 more coin-in per day. Staff to that pattern.
- **Limitation:** events are few; the model does not claim to forecast July tourism.

## Claim 6 — Spend $7,500 on Back Forty signs and hold $92,500
- **Query:** July cost = catalog unit cost after a 10-day lead. EV = 8% × theo/hr gap × hours/day × 48 machines × 20 days.
- **Result:** cost $7,500, EV $11,390, NPV $3,890. At 5% capture NPV is $-381.
- **Alternative tested:** move machines or replace them. Moves lack sample (n=4). Replacements inherit the zone.
- **Limitation:** 8% is an assumption, not a measured signage elasticity. The kill rule exists because of that.
