<?php

return [

    /*
    |--------------------------------------------------------------------------
    | Inactivity Period (days)
    |--------------------------------------------------------------------------
    |
    | The number of days after which a user is considered inactive.
    | Defaults to 7. Override via INACTIVE_USERS_DAYS in your .env file.
    |
    */
    'inactivity_days' => (int) env('INACTIVE_USERS_DAYS', 7),

    /*
    |--------------------------------------------------------------------------
    | Queue Name
    |--------------------------------------------------------------------------
    |
    | The queue on which reminder jobs are dispatched.
    | Override via INACTIVE_USERS_QUEUE in your .env file.
    |
    */
    'queue' => env('INACTIVE_USERS_QUEUE', 'reminders'),

];
