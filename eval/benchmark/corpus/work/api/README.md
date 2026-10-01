# Partner API

Authentication uses a bearer token. The rate limit is 600 requests per minute per key. When it is exceeded the API returns status 429 and the header X-RateLimit-Remaining is 0; wait for the number of seconds in Retry-After before trying again. List endpoints paginate with next_cursor. Send an idempotency key header on every POST so that retries are safe.
