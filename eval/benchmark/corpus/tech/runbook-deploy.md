# Deployment runbook

## Overview
This runbook describes how to ship the API service from a merged pull request to production. A normal deployment takes about twenty minutes and is done by the engineer who merged the change, never on Fridays after 15:00.

## Prerequisites
You need access to the container registry, the staging and production clusters, and the monitoring dashboard. Confirm that the continuous integration run on the main branch is green and that the changelog entry has been written. Check the release calendar for freezes before starting.

## Build and test
The pipeline builds the container image, runs the unit and integration test suites, and pushes the tagged image to the registry. If the integration tests fail, do not retry more than once; a repeated failure is a real failure and needs investigating before anything is deployed.

## Deploying to staging
Update the staging deployment to the new image tag and wait for all pods to report ready. Run the smoke test script, which exercises login, the main read endpoints and one write path. Staging must stay healthy for at least ten minutes before promotion.

## Promoting to production
Promote the same image tag to production using a rolling update with a maximum of one unavailable pod at a time. Announce the start in the engineering channel and keep the dashboard open on the error rate and latency panels throughout.

## Post-deploy checks
After the rollout completes, repeat the smoke test against production and check that the background workers are consuming the queue. Compare latency against the previous hour; a rise of more than twenty percent deserves a look even if no alert has fired.

## Monitoring and alerts
The error-rate alert fires when more than two percent of requests fail for five minutes. The latency alert fires when the 95th percentile exceeds 800 milliseconds. Both alerts page the on-call engineer.

## Schema changes
Changes to the database schema are shipped separately from application code, one release earlier, and must be backwards compatible so the previous version keeps working while the new one rolls out. Never drop a column in the same release that stops using it; wait one full release cycle.

## Feature flags
New behaviour is hidden behind a feature flag that is off by default. Turn the flag on for staff first, then for ten percent of traffic, then for everyone, waiting at least thirty minutes between steps and watching the dashboard.

## Communication
Post the start and the end of every production deployment in the engineering channel, with a link to the pull request and the dashboard. If customer support needs to know about a visible change, tell them before the rollout starts, not after.

## Rollback procedure
If the error rate stays above two percent for five minutes after a deployment, roll back immediately with kubectl rollout undo deployment/api and wait for the pods to become ready. Confirm in the dashboard that the error rate returns to normal, then post in the engineering channel and open an incident ticket describing what changed.

## Escalation
If the rollback does not restore service within fifteen minutes, page the engineering manager and the database on-call. Keep a written timeline of actions as you go.
