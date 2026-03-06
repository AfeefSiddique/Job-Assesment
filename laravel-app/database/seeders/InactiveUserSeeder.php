<?php

namespace Database\Seeders;

use App\Models\User;
use Carbon\Carbon;
use Illuminate\Database\Seeder;
use Illuminate\Support\Facades\Hash;

class InactiveUserSeeder extends Seeder
{
    public function run(): void
    {
        // Active user — logged in today
        User::factory()->create([
            'name'          => 'Active User',
            'email'         => 'active@example.com',
            'last_login_at' => Carbon::now(),
        ]);

        // Borderline user — logged in exactly 7 days ago (should be picked up)
        User::factory()->create([
            'name'          => 'Borderline User',
            'email'         => 'borderline@example.com',
            'last_login_at' => Carbon::now()->subDays(7),
        ]);

        // Inactive user — logged in 10 days ago
        User::factory()->create([
            'name'          => 'Inactive User',
            'email'         => 'inactive@example.com',
            'last_login_at' => Carbon::now()->subDays(10),
        ]);

        // Never-logged-in user (last_login_at is NULL)
        User::factory()->create([
            'name'          => 'Never Logged In',
            'email'         => 'never@example.com',
            'last_login_at' => null,
        ]);

        $this->command->info('Seeded 4 test users (1 active, 1 borderline, 1 inactive, 1 never-logged-in).');
    }
}
