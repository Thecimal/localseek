# Customer Portal Proposal

## Background
Customers currently email the support desk to download invoices, change their address and check order status. About forty percent of support tickets are for these three tasks. A self-service customer portal would remove most of that load and give customers an answer at any hour.

## Goals
Reduce routine support tickets by thirty percent within six months of launch, give customers a single login for all products, and cut invoice delivery time from two days to instant. Secondary goals are better data on how customers use the service and a foundation for future self-service features.

## Scope
The first release covers login, invoice download, address changes, order tracking and a contact form. Payments, subscription changes and a mobile app are explicitly out of scope and will be considered after the first release has been in use for a quarter.

## Architecture
The portal is a single-page web application backed by the existing REST services. Login uses Keystone SSO so customers need only one account across products. A thin gateway adds caching and rate limiting. Sensitive documents are served through short-lived signed links rather than public URLs.

## Phases
Phase one is discovery and design, six weeks. Phase two builds the login and invoice features, ten weeks. Phase three adds order tracking, the contact form and accessibility fixes, eight weeks. A two-week pilot with twenty friendly customers follows before general release.

## Risks
The largest risk is the data migration from the legacy system, which holds inconsistent customer records. A second risk is dependence on Keystone SSO release timing. Staffing is also tight; the plan assumes two backend engineers join in January.

## Stakeholders
The sponsor is the head of customer operations. The support desk, finance and the web team are consulted at each phase, and a monthly steering meeting reviews progress against the plan. Legal reviews the terms shown on first login.

## Success measures
Success is measured by the share of invoices downloaded through the portal, the drop in routine support tickets, the median time to complete an address change, and a short satisfaction survey shown after each task. Targets are agreed before the pilot starts.

## Support and training
Support staff receive a half-day walkthrough before the pilot and a short written guide before general release. A help page inside the portal answers the most common questions, and the contact form routes unanswered ones to the support desk.

## Budget and timeline
Total project cost is estimated at $184,000, split into $52,000 for design, $96,000 for engineering and $36,000 for testing, hosting and contingency. The first release is planned for the first quarter of next year, with the pilot in the final two weeks of the build.
