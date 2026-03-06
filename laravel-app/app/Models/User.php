<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Foundation\Auth\User as Authenticatable;
use Illuminate\Notifications\Notifiable;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Database\Eloquent\Builder;
use Carbon\Carbon;

class User extends Authenticatable
{
    use HasFactory, Notifiable;

    protected $fillable = [
        'name',
        'email',
        'password',
        'last_login_at',
    ];

    protected $hidden = [
        'password',
        'remember_token',
    ];

    protected function casts(): array
    {
        return [
            'email_verified_at' => 'datetime',
            'last_login_at'     => 'datetime',
            'password'          => 'hashed',
        ];
    }

    /**
     * Relationship: a user has many reminder logs.
     */
    public function reminderLogs(): HasMany
    {
        return $this->hasMany(ReminderLog::class);
    }

    /**
     * Scope: users inactive for more than $days days.
     * Uses the configurable inactivity_days value from config/inactive_users.php.
     */
    public function scopeInactive(Builder $query, ?int $days = null): Builder
    {
        $days      = $days ?? config('inactive_users.inactivity_days', 7);
        $threshold = Carbon::now()->subDays($days);

        return $query->where(function (Builder $q) use ($threshold) {
            // Either last_login_at is older than threshold OR it has never been set
            $q->where('last_login_at', '<=', $threshold)
              ->orWhereNull('last_login_at');
        });
    }

    /**
     * Scope: exclude users who have already received a reminder today.
     */
    public function scopeNotRemindedToday(Builder $query): Builder
    {
        $today = Carbon::today();

        return $query->whereDoesntHave('reminderLogs', function (Builder $q) use ($today) {
            $q->whereDate('sent_at', $today);
        });
    }

    /**
     * Update last_login_at to now (call this on successful login).
     */
    public function recordLogin(): void
    {
        $this->update(['last_login_at' => Carbon::now()]);
    }
}
