/**
 * DSPy Classification Studio — AngularJS 1.8 Application
 *
 * Architecture mirrors Angular 2+ patterns:
 *   - Services for API communication and state persistence
 *   - Controller organised into namespaced view-models
 *   - Component-style view sections (classify / batch / agent / graph / history)
 */
(function () {
  'use strict';

  var app = angular.module('classifierApp', ['ngAnimate']);

  /* ═══════════════════════════════════════════════════════
   * SERVICE: ApiService — all REST communication
   * ═══════════════════════════════════════════════════════ */
  app.factory('ApiService', ['$http', function ($http) {
    var API = '/api';
    return {
      health:       function ()      { return $http.get('/health'); },
      classifiers:  function ()      { return $http.get(API + '/classifiers'); },
      classify:     function (text, type) {
        return $http.post(API + '/classify', { text: text, classifier_type: type });
      },
      batchClassify: function (texts, type) {
        return $http.post(API + '/classify/batch', { texts: texts, classifier_type: type || 'sentiment' });
      },
      agentAnalyze: function (text) {
        return $http.post(API + '/agent/analyze', { text: text, enable_knowledge_graph: true });
      },
      knowledgeGraph: function () { return $http.get(API + '/knowledge-graph'); },
      seedGraph:      function () { return $http.post(API + '/knowledge-graph/seed'); },
      graphInfer:     function (q) { return $http.post(API + '/graph/infer', q); }
    };
  }]);

  /* ═══════════════════════════════════════════════════════
   * SERVICE: HistoryService — localStorage persistence
   * ═══════════════════════════════════════════════════════ */
  app.factory('HistoryService', function () {
    var KEY = 'classifier_history';
    var MAX = 50;
    return {
      load: function () {
        try { return JSON.parse(localStorage.getItem(KEY)) || []; }
        catch (e) { return []; }
      },
      save: function (list) {
        localStorage.setItem(KEY, JSON.stringify(list.slice(0, MAX)));
      },
      add: function (item) {
        var list = this.load();
        list.unshift(item);
        this.save(list);
        return list;
      },
      clear: function () {
        localStorage.removeItem(KEY);
      }
    };
  });

  /* ═══════════════════════════════════════════════════════
   * CONTROLLER: AppController
   * ═══════════════════════════════════════════════════════ */
  app.controller('AppController', [
    '$scope', 'ApiService', 'HistoryService',
    function ($scope, Api, History) {

      /* ── View state ─────────────────────────── */
      $scope.view  = 'classify';
      $scope.busy  = false;
      $scope.error = null;

      /* ── Health ─────────────────────────────── */
      $scope.health = { initialized: false, provider: '', model: '' };

      Api.health().then(function (r) {
        $scope.health = r.data;
      });

      /* ── Classify view-model ────────────────── */
      $scope.classify = { text: '', type: 'sentiment', result: null };

      $scope.doClassify = function () {
        if (!$scope.classify.text.trim()) return;
        $scope.busy = true;
        $scope.error = null;
        Api.classify($scope.classify.text, $scope.classify.type)
          .then(function (r) {
            $scope.classify.result = r.data;
            $scope.history = History.add(r.data);
          })
          .catch(errHandler)
          .finally(function () { $scope.busy = false; });
      };

      /* ── Batch view-model ───────────────────── */
      $scope.batch = { text: '', result: null };

      $scope.doBatch = function () {
        if (!$scope.batch.text.trim()) return;
        var texts = $scope.batch.text.split('\n').filter(function (t) { return t.trim(); });
        if (!texts.length) return;
        $scope.busy = true;
        $scope.error = null;
        Api.batchClassify(texts)
          .then(function (r) { $scope.batch.result = r.data; })
          .catch(errHandler)
          .finally(function () { $scope.busy = false; });
      };

      /* ── Agent view-model ───────────────────── */
      $scope.agent = { text: '', result: null };

      $scope.doAgent = function () {
        if (!$scope.agent.text.trim()) return;
        $scope.busy = true;
        $scope.error = null;
        Api.agentAnalyze($scope.agent.text)
          .then(function (r) {
            $scope.agent.result = r.data;
            $scope.history = History.add({
              classifier_type: 'agent',
              text: $scope.agent.text,
              result: r.data
            });
          })
          .catch(errHandler)
          .finally(function () { $scope.busy = false; });
      };

      /* ── Knowledge-graph view-model ─────────── */
      $scope.graph = {
        data: null,
        inference: null,
        tab: 'explore',
        search: '',
        query: { entity: '', entity_type: '', max_depth: 2, relation_filter: '' }
      };

      $scope.loadGraph = function () {
        Api.knowledgeGraph()
          .then(function (r) { $scope.graph.data = r.data; })
          .catch(function () { $scope.error = 'Failed to load knowledge graph'; });
      };

      $scope.seedGraph = function () {
        $scope.busy = true;
        $scope.error = null;
        Api.seedGraph()
          .then(function () { $scope.loadGraph(); })
          .catch(errHandler)
          .finally(function () { $scope.busy = false; });
      };

      $scope.doInfer = function () {
        var q = $scope.graph.query;
        if (!q.entity || !q.entity.trim()) {
          $scope.error = 'Entity name is required.';
          return;
        }
        $scope.busy = true;
        $scope.error = null;
        var payload = { entity: q.entity, max_depth: Number(q.max_depth || 2) };
        if (q.entity_type) payload.entity_type = q.entity_type;
        if (q.relation_filter && q.relation_filter.trim()) payload.relation_filter = q.relation_filter;

        Api.graphInfer(payload)
          .then(function (r) { $scope.graph.inference = r.data; })
          .catch(errHandler)
          .finally(function () { $scope.busy = false; });
      };

      $scope.inferEntity = function (name, type) {
        $scope.graph.query.entity = name;
        $scope.graph.query.entity_type = type || '';
        $scope.graph.tab = 'infer';
        $scope.doInfer();
      };

      /* ── Graph helpers ──────────────────────── */
      $scope.graphEntityTypes = function () {
        if (!$scope.graph.data || !$scope.graph.data.nodes) return [];
        var seen = {};
        $scope.graph.data.nodes.forEach(function (n) { seen[n.type] = true; });
        return Object.keys(seen).sort();
      };

      $scope.entitiesByType = function (t) {
        if (!$scope.graph.data || !$scope.graph.data.nodes) return [];
        return $scope.graph.data.nodes
          .filter(function (n) { return n.type === t; })
          .sort(function (a, b) { return a.name.localeCompare(b.name); });
      };

      $scope.graphRelationTypes = function () {
        if (!$scope.graph.data || !$scope.graph.data.edges) return [];
        var seen = {};
        $scope.graph.data.edges.forEach(function (e) { seen[e.relation] = true; });
        return Object.keys(seen).sort();
      };

      /* ── History ────────────────────────────── */
      $scope.history = History.load();

      $scope.clearHistory = function () {
        History.clear();
        $scope.history = [];
      };

      /* ── Shared error handler ───────────────── */
      function errHandler(err) {
        $scope.error = (err.data && err.data.error) ? err.data.error : 'Request failed';
      }
    }
  ]);

})();
