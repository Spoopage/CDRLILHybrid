using System.Collections.Generic;
using System.IO;
using Unity.AI.Navigation;
using UnityEngine;
using UnityEngine.AI;

public class MapOrganizer : MonoBehaviour
{
    [Header("Settings")]
    public string targetTag = "Obstacle";
    public string targetLayer = "Obstacle";

    [Header("Keywords Nama Objek")]
    public List<string> obstacleKeywords = new List<string> { "Wall", "Cube", "Pillar", "Box", "Statue" }; // Jika nama objek mengandung kata-kata ini, maka akan diubah

    [Header("Map Boundaries")]
    public float minX = -10f;
    public float maxX = 10f;
    public float minZ = -10f;
    public float maxZ = 10f;

    [Header("Scan Settings")]
    public float gridSize = 1.0f; // Value harus sama dengan di Agent
    public LayerMask groundLayer;
    public LayerMask obstacleLayer;

    [Header("Debug Visualization")]
    public bool showGridGizmos = true;
    public float gizmoYOffset = 0.5f; // Ketinggian visual grid dari titik nol

    private HashSet<Vector2Int> validNavMeshGrids = new HashSet<Vector2Int>();
    private HashSet<Vector2Int> goldenPathGrids = new HashSet<Vector2Int>();
    
    [Header("Auto Golden Path Generator")]
    public string rawPathFileName = "GoldenPath/GoldenPath_raw.txt";
    public int bufferRadius = 1; // 1 = melebar 1 grid ke segala arah. 2 = melebar 2 grid.
    // Asumsi: Di tab Navigation > Areas, ada indeks area tertentu.
    // Misalnya Area 0 adalah Walkable (default), dan Area 3 bernama "GoldenPath"
    // Tidak dipakai — pendekatan diganti dengan file-based golden path loading
    //public int goldenPathAreaIndex = 3;

    void Awake()
    {
        InitializeNavMeshGrids();
    }

    private void InitializeNavMeshGrids()
    {
        // 1. Ekstrak data NavMesh seperti biasa untuk mendapatkan validNavMeshGrids
        ExtractNavMeshToGrids();

        // 2. Baca file titik mentah dari rekaman ILAgent
        HashSet<Vector2Int> coreGrids = LoadRawPathFromFile();

        // 3. Terapkan algoritma pelebaran (Buffer/Dilation) sesuai ide Anda
        goldenPathGrids.Clear();

        foreach (Vector2Int coreGrid in coreGrids)
        {
            // Melakukan looping ke area sekitar grid inti berdasarkan radius buffer
            for (int x = -bufferRadius; x <= bufferRadius; x++)
            {
                for (int z = -bufferRadius; z <= bufferRadius; z++)
                {
                    Vector2Int bufferedGrid = new Vector2Int(coreGrid.x + x, coreGrid.y + z);

                    // Hanya jadikan Golden Path jika buffer berada di atas lantai yang valid (tidak menembus tembok/udara)
                    if (validNavMeshGrids.Contains(bufferedGrid))
                    {
                        goldenPathGrids.Add(bufferedGrid);
                    }
                }
            }
        }
        Debug.Log($"Golden Path di-generate secara otomatis! Grid Inti: {coreGrids.Count} -> Dilebarkan menjadi: {goldenPathGrids.Count}");
    }

    private void ExtractNavMeshToGrids()
    {
        NavMeshTriangulation navData = NavMesh.CalculateTriangulation();
        if (navData.indices.Length == 0)
        {
            Debug.LogError("NavMesh belum di-bake! Buka Window > AI > Navigation dan klik Bake.");
            return;
        }

        validNavMeshGrids.Clear();

        foreach (Vector3 v in navData.vertices)
        {
            for (float x = v.x - gridSize; x <= v.x + gridSize; x += gridSize)
            {
                for (float z = v.z - gridSize; z <= v.z + gridSize; z += gridSize)
                {
                    NavMeshHit hit;
                    if (NavMesh.SamplePosition(new Vector3(x, v.y, z), out hit, 0.5f, NavMesh.AllAreas))
                    {
                        int gx = Mathf.FloorToInt(hit.position.x / gridSize);
                        int gz = Mathf.FloorToInt(hit.position.z / gridSize);

                        // Memasukkan koordinat dasar NavMesh ke dalam memori valid
                        validNavMeshGrids.Add(new Vector2Int(gx, gz));
                    }
                }
            }
        }
    }

    private HashSet<Vector2Int> LoadRawPathFromFile()
    {
        // HashSet otomatis menyaring koordinat duplikat
        HashSet<Vector2Int> loadedGrids = new HashSet<Vector2Int>();
        string path = Path.Combine(Application.dataPath, rawPathFileName);

        if (File.Exists(path))
        {
            string[] lines = File.ReadAllLines(path);
            foreach (string line in lines)
            {
                string[] parts = line.Split(',');
                if (parts.Length == 2 && int.TryParse(parts[0], out int x) && int.TryParse(parts[1], out int z))
                {
                    loadedGrids.Add(new Vector2Int(x, z));
                }
            }
        }
        else
        {
            Debug.LogWarning($"File rute {rawPathFileName} belum ditemukan. Abaikan pesan ini jika Anda sedang dalam fase perekaman awal.");
        }
        return loadedGrids;
    }

    // Agen akan memanggil fungsi ini untuk mengecek apakah ia menginjak grid yang valid
    public bool IsValidNavMeshGrid(Vector2Int gridPosition)
    {
        return validNavMeshGrids.Contains(gridPosition);
    }

    // Agen akan memanggil ini untuk menentukan besaran reward
    public bool IsGoldenPathArea(Vector2Int gridPosition)
    {
        return goldenPathGrids.Contains(gridPosition);
    }

    public int GetTotalValidGrids()
    {
        return validNavMeshGrids.Count;
    }

    [ContextMenu("Auto Assign Obstacles")]
    public void AutoAssignObstacles()
    {
        // Mencari semua GameObject di dalam scene (termasuk yang tidak aktif)
        GameObject[] allObjects = GameObject.FindObjectsByType<GameObject>(FindObjectsInactive.Include, FindObjectsSortMode.None);
        int count = 0;

        foreach (GameObject obj in allObjects)
        {
            foreach (string keyword in obstacleKeywords)
            {
                // Mengecek apakah nama objek mengandung keyword (tidak sensitif huruf besar/kecil)
                if (obj.name.ToLower().Contains(keyword.ToLower()))
                {
                    // Update Tag
                    obj.tag = targetTag;

                    // Update Layer (menggunakan NameToLayer agar lebih aman)
                    int layerIndex = LayerMask.NameToLayer(targetLayer);
                    if (layerIndex != -1)
                    {
                        obj.layer = layerIndex;
                    }

                    count++;
                    break; // Keluar dari loop keyword jika sudah ketemu satu yang cocok
                }
            }
        }

        Debug.Log($"{count} objek telah diubah ke Layer '{targetLayer}' & Tag '{targetTag}'.");
    }


    [ContextMenu("Hitung Total Grid")]
    public void CountTotalGrids()
    {
        // Pengecekan apakah LayerMask sudah dipilih di Inspector
        if (groundLayer.value == 0)
        {
            Debug.LogError("Error: 'Ground Layer' di Inspector masih 'Nothing'");
            return;
        }

        int totalCount = 0;
        int hitCount = 0;
        int loopCount = 0;

        for (float x = minX; x <= maxX; x += gridSize)
        {
            for (float z = minZ; z <= maxZ; z += gridSize)
            {
                loopCount++;
                Vector3 startPos = new Vector3(x, 1000f, z);
                if (Physics.Raycast(startPos, Vector3.down, out RaycastHit hit, 2000f, groundLayer))
                {
                    hitCount++;
                    if (!Physics.CheckSphere(hit.point + Vector3.up * 0.5f, 0.3f, obstacleLayer))
                    {
                        totalCount++;
                        Debug.DrawLine(startPos, hit.point, Color.green, 1f);
                    }
                }
            }
        }
        Debug.Log($"Scan Raycast Selesai. Total Loop: {loopCount}, Menabrak Lantai: {hitCount}, Total Grid Valid: {totalCount}");
    }

    [ContextMenu("Hitung Grid dari NavMesh")]
    public void CalculateNavMeshGrids()
    {
        NavMeshTriangulation navData = NavMesh.CalculateTriangulation();
        if (navData.indices.Length == 0)
        {
            Debug.LogError("NavMesh belum di-bake! Buka Window > AI > Navigation dan klik Bake.");
            return;
        }

        HashSet<Vector2Int> uniqueGrids = new HashSet<Vector2Int>();

        // Melakukan sampling berdasarkan bounding box NavMesh
        foreach (Vector3 v in navData.vertices)
        {
            // Ambil area sekitar vertex untuk sampling grid
            for (float x = v.x - gridSize; x <= v.x + gridSize; x += gridSize)
            {
                for (float z = v.z - gridSize; z <= v.z + gridSize; z += gridSize)
                {
                    NavMeshHit hit;
                    if (NavMesh.SamplePosition(new Vector3(x, v.y, z), out hit, 0.5f, NavMesh.AllAreas))
                    {
                        int gx = Mathf.FloorToInt(hit.position.x / gridSize);
                        int gz = Mathf.FloorToInt(hit.position.z / gridSize);
                        uniqueGrids.Add(new Vector2Int(gx, gz));
                    }
                }
            }
        }
        Debug.Log($"Total Grid Navigasi (via NavMesh):{uniqueGrids.Count}");
    }

    // Debugging visualisasi grid di Unity Editor
    private void OnDrawGizmos()
    {
        // Cek variabel kosong untuk menghindari error
        if (!showGridGizmos || validNavMeshGrids == null || validNavMeshGrids.Count == 0) return;

        foreach (var grid in validNavMeshGrids)
        {
            // Golden Path dirender dengan warna Kuning Transparan
            // Area Eksplorasi dirender dengan warna Biru Transparan
            if (goldenPathGrids != null && goldenPathGrids.Contains(grid))
                Gizmos.color = new Color(1f, 0.92f, 0.016f, 0.5f);
            else
                Gizmos.color = new Color(0f, 0.5f, 1f, 0.5f);

            // Hitung titik tengah visual untuk dirender di Unity Editor
            Vector3 center = new Vector3((grid.x * gridSize) + (gridSize / 2f), gizmoYOffset, (grid.y * gridSize) + (gridSize / 2f));

            // Menggambar kotak grid sedikit lebih kecil dari ukuran aslinya agar terlihat pemisahnya
            Gizmos.DrawCube(center, new Vector3(gridSize * 0.9f, 0.1f, gridSize * 0.9f));
        }
    }

    [ContextMenu("Debug NavMesh Modifiers")]
    public void DebugModifiers()
    {
        NavMeshModifier[] modifiers = FindObjectsByType<NavMeshModifier>(FindObjectsSortMode.None);
        NavMeshModifierVolume[] volumes = FindObjectsByType<NavMeshModifierVolume>(FindObjectsSortMode.None);

        Debug.Log($"Modifier ditemukan: {modifiers.Length}, Volume ditemukan: {volumes.Length}");

        foreach (var m in modifiers)
            Debug.Log($"Modifier: {m.gameObject.name}, IgnoreFromBuild: {m.ignoreFromBuild}, Active: {m.gameObject.activeInHierarchy}");

        foreach (var v in volumes)
            Debug.Log($"Volume: {v.gameObject.name}, Area: {v.area}, Active: {v.gameObject.activeInHierarchy}");
    }
}