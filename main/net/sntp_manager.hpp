#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize SNTP and block until time is synchronized.
 */
void sntp_init_sync(void);

#ifdef __cplusplus
}
#endif
