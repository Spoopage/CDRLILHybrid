using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using System.IO;

public class HybridAgent : Agent
{
    [Header("Movement Settings")]
    public float walkSpeed = 5f;
    public float sprintSpeed = 8f;
    public float rotationSpeed = 150f;
    public float groundDrag = 5f;

    [Header("Jumping")]
    public float jumpForce = 7f;
    public float jumpCooldown = 0.25f;
    public float airMultiplier = 0.4f;
    private bool readyToJump = true;

    [Header("Physics & Environment")]
    public float playerHeight = 2f;
    public LayerMask whatIsGround;
    private bool grounded;
    private Rigidbody rb;

    [Header("Coverage Tracking")]
    public float gridCellSize = 1f;

    // Memisahkan memori untuk area jalur utama dan area eksplorasi
    private HashSet<Vector2Int> goldenPathGrids = new HashSet<Vector2Int>();
    private HashSet<Vector2Int> explorationGrids = new HashSet<Vector2Int>();

    // Tidak dipakai — tidak di-log dan reset-nya di-comment di OnEpisodeBegin
    //private int totalGoldenPathGrids = 0;
    //private int totalExplorationGrids = 0;

    [Header("Bug Reporting")]
    public string fileName = "BugReport_Hybrid.csv";
    public float stuckTimeout = 5f;
    private float stuckTimer = 0f;
    private Vector3 lastPosition;

    [Header("Statistics")]
    private int jumpCount, sprintCount, totalActions;
    private int episodeCount = 0;

    [Header("Coverage Export")]
    public string coverageFileName = "CoverageData_Hybrid.csv";

    [Header("Map References")]
    public MapOrganizer mapOrganizer;

    [Header("Debug System")]
    public bool enableDebugLog = false;
    public bool enableVisualDebug = true;

    [Header("Physics Detection")]
    public float physicsVelocityThreshold = 300f;
    public float physicsVelocityDuration = 0.1f; // harus melampaui threshold selama 0.1 detik
    private float physicsVelocityTimer = 0f;

    [Header("Position Logger")]
    [Tooltip("Nama file CSV untuk log posisi spasial setiap 5 frame.")]
    public string positionFileName = "PositionLog_Hybrid_MapB.csv";
    [Tooltip("Catat posisi setiap N frame. Default 5 sesuai metodologi skripsi.")]
    public int positionLogInterval = 5;
    private StreamWriter positionWriter;
    private int frameCounter = 0;

    private Transform spawnPoint;

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
        rb.freezeRotation = true;
        spawnPoint = GameObject.Find("SpawnPoint")?.transform;
        InitPositionLogger();
    }
    void Update()
    {
        // Catat posisi setiap positionLogInterval frame
        frameCounter++;
        if (frameCounter % positionLogInterval == 0)
        {
            LogPosition();
        }
    }

    void OnDestroy()
    {
        ClosePositionLogger();
    }

    // -----------------------------------------------------------------------
    // EPISODE
    // -----------------------------------------------------------------------
    public override void OnEpisodeBegin()
    {
        episodeCount++;

        // Mengirimkan statistik aksi ke TensorBoard
        if (totalActions > 0)
        {
            Academy.Instance.StatsRecorder.Add("Actions/JumpUsage", (float)jumpCount / totalActions);
            Academy.Instance.StatsRecorder.Add("Actions/SprintUsage", (float)sprintCount / totalActions);
        }

        int goldenPathCount = goldenPathGrids.Count;
        int explorationCount = explorationGrids.Count;
        float totalCoverage = goldenPathCount + explorationCount;

        // Mengirimkan statistik persentase eksplorasi ke TensorBoard sebelum memori direset
        if (totalCoverage > 0)
        {
            //Academy.Instance.StatsRecorder.Add("Coverage/GoldenPath_Count", goldenPathGrids.Count);
            //Academy.Instance.StatsRecorder.Add("Coverage/Exploration_Count", explorationGrids.Count);
            //Academy.Instance.StatsRecorder.Add("Coverage/Total_Coverage", totalCoverage);
            Academy.Instance.StatsRecorder.Add("Grid_Exploration/Golden_Path_Count", goldenPathCount);
            Academy.Instance.StatsRecorder.Add("Grid_Exploration/New_Area_Count", explorationCount);
            Academy.Instance.StatsRecorder.Add("Grid_Exploration/Total_Coverage", totalCoverage);

            // Rasio area non-Golden Path yang berhasil dicapai
            //float totalCoverage = goldenPathCount + explorationCount;
            float explorationPercentage = ((float)explorationCount / totalCoverage) * 100f;
            Academy.Instance.StatsRecorder.Add("Exploration_Ratio/New_Area_Percentage", explorationPercentage);
        }

        // Save koordinat ke CSV
        SaveCoverageToFile();

        // Reset semua state dan memori
        goldenPathGrids.Clear();
        explorationGrids.Clear();
        goldenPathGrids.TrimExcess();
        explorationGrids.TrimExcess();

        jumpCount = 0; sprintCount = 0; totalActions = 0;
        //totalGoldenPathGrids = 0; totalExplorationGrids = 0;
        stuckTimer = 0f;

        // Relokasi ke Spawn Point
        if (spawnPoint)
        {
            transform.position = spawnPoint.position;
            transform.rotation = spawnPoint.rotation;
        }

        rb.linearVelocity = Vector3.zero;
    }

    // -----------------------------------------------------------------------
    // OBSERVATIONS & ACTIONS
    // -----------------------------------------------------------------------
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(transform.forward);
        sensor.AddObservation(rb.linearVelocity);
        sensor.AddObservation(rb.angularVelocity.y);
        sensor.AddObservation(grounded ? 1.0f : 0.0f);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        totalActions++;
        DetectAnomalies();
        ProcessMovement(actions);
        TrackGridCoverage();
    }

    private void ProcessMovement(ActionBuffers actions)
    {
        int moveZ = actions.DiscreteActions[0];
        int rotateY = actions.DiscreteActions[1];
        bool isSprinting = actions.DiscreteActions[2] == 1;
        bool isJumping = actions.DiscreteActions[3] == 1;

        if (isSprinting) sprintCount++;

        // Rotasi
        float rotDir = rotateY == 1 ? 1f : (rotateY == 2 ? -1f : 0f);
        transform.Rotate(Vector3.up, rotDir * rotationSpeed * Time.fixedDeltaTime);

        // Gerak
        float moveInput = moveZ == 1 ? 1f : (moveZ == 2 ? -1f : 0f);
        float speed = isSprinting ? sprintSpeed : walkSpeed;
        Vector3 moveDir = transform.forward * moveInput;

        grounded = Physics.Raycast(transform.position, Vector3.down, playerHeight * 0.5f + 0.75f, whatIsGround)
                   && !FallZoneTrigger.IsInFallZone(GetComponent<Collider>());
        rb.linearDamping = grounded ? groundDrag : 0f;

        float forceMult = grounded ? 10f : 10f * airMultiplier;
        rb.AddForce(moveDir.normalized * speed * forceMult, ForceMode.Force);

        // Speed Limiter
        Vector3 flatVel = new Vector3(rb.linearVelocity.x, 0f, rb.linearVelocity.z);

        if (flatVel.magnitude > speed)
        {
            Vector3 limitedVel = flatVel.normalized * speed;
            rb.linearVelocity = new Vector3(limitedVel.x, rb.linearVelocity.y, limitedVel.z);
        }

        // Lompat
        if (isJumping && readyToJump && grounded)
        {
            jumpCount++;
            readyToJump = false;
            rb.linearVelocity = new Vector3(rb.linearVelocity.x, 0f, rb.linearVelocity.z);
            rb.AddForce(transform.up * jumpForce, ForceMode.Impulse);
            Invoke(nameof(ResetJump), jumpCooldown);
        }
    }

    // -----------------------------------------------------------------------
    // COVERAGE
    // -----------------------------------------------------------------------
    private void TrackGridCoverage()
    {
        int gx = Mathf.FloorToInt(transform.position.x / gridCellSize);
        int gz = Mathf.FloorToInt(transform.position.z / gridCellSize);
        Vector2Int currentGrid = new Vector2Int(gx, gz);

        // Query apakah agen berdiri di koordinat NavMesh yang valid
        if (mapOrganizer.IsValidNavMeshGrid(currentGrid))
        {
            // Jika valid, cek lagi apakah itu di atas Golden Path
            if (mapOrganizer.IsGoldenPathArea(currentGrid))
            {
                if (goldenPathGrids.Add(currentGrid))
                {
                    //totalGoldenPathGrids++;
                    // Reward dasar
                    AddReward(0.1f);

                    if (enableDebugLog)
                        Debug.Log($"<color=yellow>[REWARD +0.1]</color> Agen menemukan Grid Golden Path baru: {currentGrid}");
                }
            }
            else
            {
                // Jika itu di Golden Path, berikan penekanan eksplorasi
                if (explorationGrids.Add(currentGrid))
                {
                    //totalExplorationGrids++;
                    // Reward yang 3x lebih besar untuk memaksa agen keluar dari Golden Path
                    AddReward(0.3f);

                    if (enableDebugLog)
                        Debug.Log($"<color=cyan>[REWARD +0.3]</color> Agen mengeksplorasi area bebas di Grid: {currentGrid}");
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // ANOMALY DETECTION
    // -----------------------------------------------------------------------
    private void DetectAnomalies()
    {
        // Deteksi Anomali Navigasi (Stuck)
        if (Vector3.Distance(transform.position, lastPosition) < 0.2f)
            stuckTimer += Time.fixedDeltaTime;
        else
        {
            stuckTimer = 0f;
            lastPosition = transform.position;
        }

        if (stuckTimer > stuckTimeout)
        {
            ReportBug("NavigationBug_Stuck");
            // PENALTI: Agen dihukum karena membuang-buang waktu simulasi
            SetReward(-1f);

            if (enableDebugLog)
                Debug.LogError($"<color=red>[PENALTY -1.0]</color> Navigation Bug! Agen stuck di posisi {transform.position}. Episode Berakhir.");

            EndEpisode();
        }

        // Deteksi Anomali Geometri (Jatuh keluar peta)
        if (transform.position.y < -10f)
        {
            ReportBug("GeometryBug_FallThroughFloor");
            // Penalti diperlukan untuk Hybrid: komponen BC mendorong agen ke jalur demonstrasi
            // yang mungkin melewati area rawan fall-through. Tanpa penalti, policy kolaps ke
            // loop spawn->jatuh setelah BC memudar (terbukti di Hybrid_Retraining_01).
            SetReward(-1f);

            if (enableDebugLog)
                Debug.LogError($"Geometry Bug! Agen jatuh di koordinat {transform.position.x}, {transform.position.z}. Episode Berakhir.");

            EndEpisode();
        }

        // Deteksi Anomali Fisika (Lonjakan akselerasi kinetik)
        if (rb.linearVelocity.magnitude > physicsVelocityThreshold)
        {
            physicsVelocityTimer += Time.fixedDeltaTime;
            if (physicsVelocityTimer >= physicsVelocityDuration)
            {
                ReportBug("PhysicsBug_VelocityExplosion");
                SetReward(-1f);
                EndEpisode();
                physicsVelocityTimer = 0f;
            }
        }
        else
        {
            physicsVelocityTimer = 0f;
        }
    }

    private void ReportBug(string type)
    {
        try
        {
            string path = Path.Combine(Application.dataPath, fileName);
            string row = $"{System.DateTime.Now:HH:mm:ss},{type},{transform.position.x:F2},{transform.position.y:F2},{transform.position.z:F2},{CompletedEpisodes}\n";
            if (!File.Exists(path)) File.WriteAllText(path, "Time,Type,X,Y,Z,Episode\n");
            File.AppendAllText(path, row);
        }
        catch { /* Mengabaikan file lock saat direkam bersamaan oleh banyak agen */ }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<int> discreteActions = actionsOut.DiscreteActions;

        discreteActions[0] = 0;
        discreteActions[1] = 0;
        discreteActions[2] = 0;
        discreteActions[3] = 0;

        if (Input.GetKey(KeyCode.W)) discreteActions[0] = 1;
        else if (Input.GetKey(KeyCode.S)) discreteActions[0] = 2;

        if (Input.GetKey(KeyCode.D)) discreteActions[1] = 1;
        else if (Input.GetKey(KeyCode.A)) discreteActions[1] = 2;

        if (Input.GetKey(KeyCode.LeftShift)) discreteActions[2] = 1;
        if (Input.GetKey(KeyCode.Space)) discreteActions[3] = 1;
    }

    private void ResetJump() => readyToJump = true;

    private void SaveCoverageToFile()
    {
        if (goldenPathGrids.Count == 0 && explorationGrids.Count == 0) return;

        string path = Path.Combine(Application.dataPath, coverageFileName);

        // Pengecekan apakah file baru pertama kali dibuat
        bool isNewFile = !File.Exists(path);

        using (StreamWriter writer = new StreamWriter(path, true))
        {
            // Jika ini file baru, tulis judul kolom terlebih dahulu di baris pertama
            if (isNewFile)
            {
                writer.WriteLine("Episode,X,Z,AreaType");
            }

            // Menyimpan koordinat jalur utama
            foreach (var grid in goldenPathGrids)
            {
                writer.WriteLine($"{episodeCount},{grid.x},{grid.y},GoldenPath");
            }

            // Menyimpan koordinat area eksplorasi baru
            foreach (var grid in explorationGrids)
            {
                writer.WriteLine($"{episodeCount},{grid.x},{grid.y},Exploration");
            }
        }
    }

    private void OnDrawGizmos()
    {
        // Fitur ini hanya menyala saat game berjalan dan opsi visual dinyalakan
        if (!Application.isPlaying || !enableVisualDebug) return;

        // Menggambar area Golden Path yang berhasil diinjak (Warna Hijau Transparan)
        if (goldenPathGrids != null)
        {
            Gizmos.color = new Color(0f, 1f, 0f, 0.4f);
            foreach (var grid in goldenPathGrids)
            {
                Vector3 pos = new Vector3(grid.x * gridCellSize, 0.1f, grid.y * gridCellSize);
                Gizmos.DrawCube(pos, new Vector3(gridCellSize, 0.1f, gridCellSize));
            }
        }

        // Menggambar area Eksplorasi baru yang berhasil diinjak (Warna Cyan Transparan)
        if (explorationGrids != null)
        {
            Gizmos.color = new Color(0f, 1f, 1f, 0.4f);
            foreach (var grid in explorationGrids)
            {
                Vector3 pos = new Vector3(grid.x * gridCellSize, 0.1f, grid.y * gridCellSize);
                Gizmos.DrawCube(pos, new Vector3(gridCellSize, 0.1f, gridCellSize));
            }
        }
    }

    // -----------------------------------------------------------------------
    // POSITION LOGGER
    // -----------------------------------------------------------------------

    private void InitPositionLogger()
    {
        try
        {
            string path = Path.Combine(Application.dataPath, positionFileName);
            bool isNewFile = !File.Exists(path);
            positionWriter = new StreamWriter(path, append: true);
            if (isNewFile)
                positionWriter.WriteLine("Episode,Frame,X,Y,Z");
            positionWriter.AutoFlush = true;
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"[HybridAgent] Gagal membuka position logger: {e.Message}");
        }
    }

    private void LogPosition()
    {
        if (positionWriter == null) return;
        try
        {
            Vector3 pos = transform.position;
            positionWriter.WriteLine(
                $"{CompletedEpisodes},{frameCounter},{pos.x:F2},{pos.y:F2},{pos.z:F2}");
        }
        catch { /* Ignore file lock */ }
    }

    private void ClosePositionLogger()
    {
        try
        {
            positionWriter?.Flush();
            positionWriter?.Close();
            positionWriter = null;
        }
        catch { /* Ignore */ }
    }
}